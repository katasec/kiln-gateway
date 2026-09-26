package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/katasec/kiln"
	"github.com/katasec/kiln/message"
)

// This file implements the subset of the OpenAI Responses API that lets a host
// CLI which speaks Responses (e.g. Codex) point at a kiln agent. Like the
// Chat Completions path, the Engine owns the loop: we map model -> agent, run
// the agent, and return the final assistant text. Inbound tools and the host's
// own instructions are not yet relayed (that is the passthrough / graduation
// work); the agent's scaffold remains the authoritative system prompt.

// --- request ---

type responsesRequest struct {
	Model        string         `json:"model"`
	Instructions string         `json:"instructions,omitempty"`
	Input        responsesInput `json:"input"`
	Stream       bool           `json:"stream,omitempty"`
}

// responsesInput decodes the Responses "input" field, which may be a bare
// string or an array of items (messages, plus tool/reasoning items we skip).
type responsesInput struct {
	messages []kiln.Message
}

func (ri *responsesInput) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		ri.messages = []kiln.Message{message.UserText(s)}
		return nil
	}
	var items []struct {
		Type    string       `json:"type"`
		Role    string       `json:"role"`
		Content contentField `json:"content"`
	}
	if err := json.Unmarshal(b, &items); err != nil {
		// Tolerate unexpected shapes rather than failing the whole request; the
		// handler logs the raw body so the parser can be tightened to match.
		return nil
	}
	for _, it := range items {
		// Keep message items (type "message" or unset); skip function_call,
		// function_call_output, reasoning, etc.
		if it.Type != "" && it.Type != "message" {
			continue
		}
		if it.Content.text == "" {
			continue
		}
		ri.messages = append(ri.messages, kiln.Message{
			Role:    toKilnRole(it.Role),
			Content: []kiln.ContentBlock{message.Text(it.Content.text)},
		})
	}
	return nil
}

// --- response object (shared by JSON reply and streaming events) ---

type responsesResponse struct {
	ID        string          `json:"id"`
	Object    string          `json:"object"`
	CreatedAt int64           `json:"created_at"`
	Status    string          `json:"status"`
	Model     string          `json:"model"`
	Output    []responsesItem `json:"output"`
	Usage     *responsesUsage `json:"usage,omitempty"`
	Error     any             `json:"error"`
}

type responsesItem struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"`
	Status  string          `json:"status"`
	Role    string          `json:"role"`
	Content []responsesPart `json:"content"`
}

type responsesPart struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func (g *Gateway) handleResponses(w http.ResponseWriter, r *http.Request) {
	data, err := readBody(r)
	if err != nil {
		g.writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}

	var req responsesRequest
	if err := json.Unmarshal(data, &req); err != nil {
		g.logger.Printf("responses: decode error: %v; raw=%s", err, truncate(data, 2000))
		g.writeError(w, http.StatusBadRequest, "invalid_request_error", "could not parse request body")
		return
	}

	g.logger.Printf("responses: model=%q stream=%v messages=%d", req.Model, req.Stream, len(req.Input.messages))
	if len(req.Input.messages) == 0 {
		// We accepted the body but extracted no user content — log the raw
		// payload so the parser can be matched to Codex's exact schema.
		g.logger.Printf("responses: WARNING parsed 0 messages; raw=%s", truncate(data, 2000))
	}

	agent, resolved, ok := g.resolve(req.Model)
	if !ok {
		g.writeError(w, http.StatusNotFound, "invalid_request_error",
			fmt.Sprintf("model %q does not exist", req.Model))
		return
	}
	if resolved != req.Model {
		g.logger.Printf("responses: model %q -> agent %q (default)", req.Model, resolved)
	}

	resp, err := agent.Run(r.Context(), kiln.AgentRequest{Messages: req.Input.messages})
	if err != nil {
		g.writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}

	id := "resp_" + uuid.NewString()
	itemID := "msg_" + uuid.NewString()
	created := time.Now().Unix()
	text := resp.LastText()
	usage := &responsesUsage{
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		TotalTokens:  totalTokens(resp.Usage),
	}

	completed := responsesResponse{
		ID:        id,
		Object:    "response",
		CreatedAt: created,
		Status:    "completed",
		Model:     req.Model,
		Output: []responsesItem{{
			ID:      itemID,
			Type:    "message",
			Status:  "completed",
			Role:    "assistant",
			Content: []responsesPart{{Type: "output_text", Text: text, Annotations: []any{}}},
		}},
		Usage: usage,
	}

	if req.Stream {
		g.writeResponsesStream(w, completed, itemID, text)
		return
	}
	g.writeJSON(w, http.StatusOK, completed)
}

// writeResponsesStream emits the Responses streaming protocol: named SSE events
// terminating in response.completed. The Engine runs to completion first, so
// the text is delivered as a single output_text delta rather than token by
// token; the event envelope is what Codex needs to consume a streamed turn.
func (g *Gateway) writeResponsesStream(w http.ResponseWriter, completed responsesResponse, itemID, text string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		g.writeError(w, http.StatusInternalServerError, "server_error", "streaming unsupported")
		return
	}

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	seq := 0
	send := func(eventType string, data map[string]any) {
		data["type"] = eventType
		data["sequence_number"] = seq
		seq++
		b, err := json.Marshal(data)
		if err != nil {
			g.logger.Printf("responses stream: marshal %s event: %v", eventType, err)
			return
		}
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, b)
		flusher.Flush()
	}

	// In-progress snapshot: same response, empty output, status in_progress.
	inProgress := completed
	inProgress.Status = "in_progress"
	inProgress.Output = []responsesItem{}
	inProgress.Usage = nil

	emptyItem := responsesItem{ID: itemID, Type: "message", Status: "in_progress", Role: "assistant", Content: []responsesPart{}}
	finalItem := completed.Output[0]

	send("response.created", map[string]any{"response": inProgress})
	send("response.in_progress", map[string]any{"response": inProgress})
	send("response.output_item.added", map[string]any{"output_index": 0, "item": emptyItem})
	send("response.content_part.added", map[string]any{
		"item_id": itemID, "output_index": 0, "content_index": 0,
		"part": responsesPart{Type: "output_text", Text: "", Annotations: []any{}},
	})
	send("response.output_text.delta", map[string]any{
		"item_id": itemID, "output_index": 0, "content_index": 0, "delta": text,
	})
	send("response.output_text.done", map[string]any{
		"item_id": itemID, "output_index": 0, "content_index": 0, "text": text,
	})
	send("response.content_part.done", map[string]any{
		"item_id": itemID, "output_index": 0, "content_index": 0,
		"part": responsesPart{Type: "output_text", Text: text, Annotations: []any{}},
	})
	send("response.output_item.done", map[string]any{"output_index": 0, "item": finalItem})
	send("response.completed", map[string]any{"response": completed})
}

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "…(truncated)"
}
