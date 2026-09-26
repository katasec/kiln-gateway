// Package gateway exposes kiln Agents behind an OpenAI-compatible HTTP API.
//
// The Gateway implements the minimal surface needed to point an existing OpenAI
// client (set OPENAI_BASE_URL) at a kiln agent: GET /v1/models lists the
// available agents as models, and POST /v1/chat/completions runs the named
// agent's full loop (the Engine owns the loop) and returns an OpenAI chat
// completion.
//
// The agent name is the OpenAI "model" field. The Gateway is stateless: an
// OpenAI client sends the full message history on every call, so the agents it
// serves should be created with kiln.Config{DisableMemory: true}.
//
// Agents are dependencies of the Gateway, not servers themselves; the Gateway
// owns the HTTP lifecycle and routes requests to the selected kiln Agent.
package gateway

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/katasec/kiln"
	"github.com/katasec/kiln/message"
	"github.com/klauspost/compress/zstd"
)

// Config holds the dependencies and settings for a Gateway. It is the single
// place dependencies are injected.
type Config struct {
	// Addr is the listen address for the HTTP server (e.g. ":8787").
	Addr string
	// Agents maps the model name a client requests to the agent that serves it.
	Agents map[string]*kiln.Agent
	// DefaultAgent names the agent used when a client requests an unknown model
	// id. Empty means strict mode (unknown model -> 404).
	DefaultAgent string
	// Logger receives request and error logs. Defaults to log.Default() if nil.
	Logger *log.Logger
}

// Gateway routes OpenAI-compatible requests to kiln Agents keyed by model name
// and owns the HTTP server lifecycle.
type Gateway struct {
	agents       map[string]*kiln.Agent
	defaultAgent string
	logger       *log.Logger
	mux          *http.ServeMux
	httpServer   *http.Server
}

// New wires a Gateway from cfg. The map key in cfg.Agents is the model name a
// client requests (e.g. "tempered_reviewer").
func New(cfg Config) *Gateway {
	logger := cfg.Logger
	if logger == nil {
		logger = log.Default()
	}
	g := &Gateway{
		agents:       cfg.Agents,
		defaultAgent: cfg.DefaultAgent,
		logger:       logger,
		mux:          http.NewServeMux(),
	}
	g.routes()
	g.httpServer = &http.Server{Addr: cfg.Addr, Handler: g}
	return g
}

// routes declares every endpoint the Gateway serves, in one place.
func (g *Gateway) routes() {
	g.mux.HandleFunc("GET /v1/models", g.handleModels)
	g.mux.HandleFunc("POST /v1/chat/completions", g.handleChatCompletions)
	g.mux.HandleFunc("POST /v1/responses", g.handleResponses)
}

// Start runs the HTTP server and blocks until it stops. A clean shutdown via
// Stop returns nil rather than http.ErrServerClosed. Process lifecycle and
// signal handling are the caller's responsibility.
func (g *Gateway) Start(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	err := g.httpServer.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

// Stop gracefully shuts down the HTTP server, waiting for in-flight requests to
// finish within ctx's deadline.
func (g *Gateway) Stop(ctx context.Context) error {
	return g.httpServer.Shutdown(ctx)
}

// resolve maps a requested model name to an agent, falling back to the
// configured default agent for unrecognized model ids. It returns the agent and
// the resolved agent name, or ok=false when no agent applies.
func (g *Gateway) resolve(model string) (*kiln.Agent, string, bool) {
	if a, ok := g.agents[model]; ok {
		return a, model, true
	}
	if g.defaultAgent != "" {
		if a, ok := g.agents[g.defaultAgent]; ok {
			return a, g.defaultAgent, true
		}
	}
	return nil, "", false
}

// ServeHTTP implements http.Handler. It logs each request's method, path, and
// status so the serve path is debuggable when a host CLI (e.g. Codex) points at
// it — an unexpected 404 on /v1/responses immediately shows a wire-API mismatch.
func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
	g.mux.ServeHTTP(sw, r)
	g.logger.Printf("%s %s -> %d", r.Method, r.URL.Path, sw.status)
}

// statusWriter captures the response status for logging while preserving the
// http.Flusher behaviour the streaming path depends on.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// readBody reads and decompresses an HTTP request body. Host CLIs (notably
// Codex) send the request zstd- or gzip-compressed; it honours Content-Encoding
// and falls back to sniffing the magic bytes when the header is absent.
func readBody(r *http.Request) ([]byte, error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}

	enc := strings.ToLower(strings.TrimSpace(r.Header.Get("Content-Encoding")))
	if enc == "" {
		switch {
		case len(raw) >= 4 && raw[0] == 0x28 && raw[1] == 0xb5 && raw[2] == 0x2f && raw[3] == 0xfd:
			enc = "zstd"
		case len(raw) >= 2 && raw[0] == 0x1f && raw[1] == 0x8b:
			enc = "gzip"
		}
	}

	switch enc {
	case "zstd":
		zr, err := zstd.NewReader(bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		defer zr.Close()
		return io.ReadAll(zr)
	case "gzip":
		gz, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		defer gz.Close()
		return io.ReadAll(gz)
	case "deflate":
		fr := flate.NewReader(bytes.NewReader(raw))
		defer fr.Close()
		return io.ReadAll(fr)
	default:
		return raw, nil
	}
}

// --- OpenAI wire types (request) ---

type chatRequest struct {
	Model    string       `json:"model"`
	Messages []reqMessage `json:"messages"`
	Stream   bool         `json:"stream,omitempty"`
}

type reqMessage struct {
	Role    string       `json:"role"`
	Content contentField `json:"content"`
}

// contentField decodes message content that may be either a plain string or an
// array of typed parts. It covers both Chat Completions parts
// ([{type:"text",text:"..."}]) and Responses parts
// ([{type:"input_text"|"output_text",text:"..."}]) by concatenating every
// part's text regardless of part type.
type contentField struct {
	text string
}

func (c *contentField) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		c.text = s
		return nil
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(b, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			sb.WriteString(p.Text)
		}
		c.text = sb.String()
		return nil
	}
	// Tolerate unexpected shapes (objects, nulls, non-text parts) rather than
	// failing the whole request; best-effort text extraction only.
	return nil
}

// --- OpenAI wire types (response) ---

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []respChoice `json:"choices"`
	Usage   usage        `json:"usage"`
}

type respChoice struct {
	Index        int         `json:"index"`
	Message      respMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type respMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// --- streaming wire types ---

type streamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
}

type streamChoice struct {
	Index        int    `json:"index"`
	Delta        delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// --- models endpoint ---

type modelList struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (g *Gateway) handleModels(w http.ResponseWriter, _ *http.Request) {
	now := time.Now().Unix()
	list := modelList{Object: "list"}
	for name := range g.agents {
		list.Data = append(list.Data, modelInfo{
			ID:      name,
			Object:  "model",
			Created: now,
			OwnedBy: "kiln",
		})
	}
	g.writeJSON(w, http.StatusOK, list)
}

func (g *Gateway) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	data, err := readBody(r)
	if err != nil {
		g.writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}
	var req chatRequest
	if err := json.Unmarshal(data, &req); err != nil {
		g.writeError(w, http.StatusBadRequest, "invalid_request_error", "could not parse request body")
		return
	}

	g.logger.Printf("chat: model=%q stream=%v messages=%d", req.Model, req.Stream, len(req.Messages))

	agent, resolved, ok := g.resolve(req.Model)
	if !ok {
		g.writeError(w, http.StatusNotFound, "invalid_request_error",
			fmt.Sprintf("model %q does not exist", req.Model))
		return
	}
	if resolved != req.Model {
		g.logger.Printf("chat: model %q -> agent %q (default)", req.Model, resolved)
	}

	resp, err := agent.Run(r.Context(), kiln.AgentRequest{
		Messages: translateMessages(req.Messages),
	})
	if err != nil {
		g.writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}

	id := "chatcmpl-" + uuid.NewString()
	created := time.Now().Unix()
	text := resp.LastText()
	finish := toOAIFinish(resp.FinishReason)

	if req.Stream {
		g.writeStream(w, id, created, req.Model, text, finish)
		return
	}

	g.writeJSON(w, http.StatusOK, chatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   req.Model,
		Choices: []respChoice{{
			Index:        0,
			Message:      respMessage{Role: "assistant", Content: text},
			FinishReason: finish,
		}},
		Usage: usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      totalTokens(resp.Usage),
		},
	})
}

// translateMessages maps OpenAI messages to kiln messages 1:1. The agent's own
// scaffold remains the authoritative system prompt; any client-supplied system
// message flows through as a kiln system message.
func translateMessages(msgs []reqMessage) []kiln.Message {
	out := make([]kiln.Message, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, kiln.Message{
			Role:    toKilnRole(m.Role),
			Content: []kiln.ContentBlock{message.Text(m.Content.text)},
		})
	}
	return out
}

func toKilnRole(r string) kiln.Role {
	switch r {
	case "system":
		return kiln.RoleSystem
	case "assistant":
		return kiln.RoleAssistant
	case "tool":
		return kiln.RoleTool
	default:
		return kiln.RoleUser
	}
}

func toOAIFinish(r kiln.FinishReason) string {
	switch r {
	case kiln.FinishReasonIterLimit:
		return "length"
	default:
		// stop, error, and tool_use (already resolved by the loop) all present
		// as a completed turn to the client.
		return "stop"
	}
}

func totalTokens(u kiln.TokenUsage) int {
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.InputTokens + u.OutputTokens
}

// writeStream emits the completed response as SSE chat.completion.chunk frames.
//
// The Engine runs the loop to completion before this is called, so the content
// is delivered in a single delta rather than token-by-token. The endpoint still
// speaks SSE so that clients requiring stream:true work; true token streaming
// depends on a streaming Provider, which the Engine does not yet expose.
func (g *Gateway) writeStream(w http.ResponseWriter, id string, created int64, model, text, finish string) {
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

	send := func(c streamChunk) {
		b, err := json.Marshal(c)
		if err != nil {
			g.logger.Printf("chat stream: marshal chunk: %v", err)
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	base := func() streamChunk {
		return streamChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: model}
	}

	role := base()
	role.Choices = []streamChoice{{Index: 0, Delta: delta{Role: "assistant"}}}
	send(role)

	content := base()
	content.Choices = []streamChoice{{Index: 0, Delta: delta{Content: text}}}
	send(content)

	stop := base()
	stop.Choices = []streamChoice{{Index: 0, Delta: delta{}, FinishReason: finish}}
	send(stop)

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (g *Gateway) writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		g.logger.Printf("writeJSON: encode response: %v", err)
	}
}

func (g *Gateway) writeError(w http.ResponseWriter, status int, errType, msg string) {
	g.writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    errType,
		},
	})
}
