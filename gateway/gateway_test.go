package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/katasec/kiln"
	"github.com/katasec/kiln/message"
)

// recordingProvider is a stub kiln.Provider that returns a canned assistant
// reply and records the request it was given, so tests can prove the HTTP layer
// translated and forwarded correctly.
type recordingProvider struct {
	reply    string
	lastReq  kiln.ProviderRequest
	gotCalls int
}

func (p *recordingProvider) Generate(_ context.Context, req kiln.ProviderRequest) (*kiln.ProviderResponse, error) {
	p.lastReq = req
	p.gotCalls++
	return &kiln.ProviderResponse{
		Messages:     []kiln.Message{message.AssistantText(p.reply)},
		FinishReason: kiln.FinishReasonStop,
		Usage:        kiln.TokenUsage{InputTokens: 11, OutputTokens: 7},
	}, nil
}

func newTestServer(t *testing.T, prov kiln.Provider) *Gateway {
	t.Helper()
	agent, err := kiln.NewAgent(kiln.Config{
		Provider:      prov,
		SystemPrompt:  "SCAFFOLD",
		DisableMemory: true,
	})
	if err != nil {
		t.Fatalf("NewAgent: %v", err)
	}
	return New(Config{Agents: map[string]*kiln.Agent{"forged_reviewer": agent}})
}

// newTestServerWithDefault is like newTestServer but configures a default agent
// so unknown model ids fall back instead of 404ing.
func newTestServerWithDefault(t *testing.T, prov kiln.Provider, def string) *Gateway {
	t.Helper()
	agent, err := kiln.NewAgent(kiln.Config{
		Provider:      prov,
		SystemPrompt:  "SCAFFOLD",
		DisableMemory: true,
	})
	if err != nil {
		t.Fatalf("NewAgent: %v", err)
	}
	return New(Config{Agents: map[string]*kiln.Agent{"forged_reviewer": agent}, DefaultAgent: def})
}

func TestModelsListsAgents(t *testing.T) {
	srv := newTestServer(t, &recordingProvider{reply: "ok"})

	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var list modelList
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if list.Object != "list" || len(list.Data) != 1 || list.Data[0].ID != "forged_reviewer" {
		t.Fatalf("unexpected model list: %+v", list)
	}
}

func TestChatCompletionsRunsAgent(t *testing.T) {
	prov := &recordingProvider{reply: "a grounded review"}
	srv := newTestServer(t, prov)

	body := `{"model":"forged_reviewer","messages":[{"role":"user","content":"review this repo"}]}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var resp chatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Object != "chat.completion" {
		t.Errorf("object = %q", resp.Object)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "a grounded review" {
		t.Fatalf("unexpected choices: %+v", resp.Choices)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", resp.Choices[0].FinishReason)
	}
	if resp.Usage.PromptTokens != 11 || resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Errorf("usage = %+v", resp.Usage)
	}

	// Prove the loop actually ran and translation reached the provider.
	if prov.gotCalls != 1 {
		t.Errorf("provider calls = %d, want 1", prov.gotCalls)
	}
	if prov.lastReq.SystemPrompt != "SCAFFOLD" {
		t.Errorf("scaffold system prompt not forwarded: %q", prov.lastReq.SystemPrompt)
	}
	last := prov.lastReq.Messages[len(prov.lastReq.Messages)-1]
	if last.Text() != "review this repo" {
		t.Errorf("user message not forwarded: %q", last.Text())
	}
}

func TestChatCompletionsUnknownModel(t *testing.T) {
	srv := newTestServer(t, &recordingProvider{reply: "ok"})

	body := `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
}

func TestChatCompletionsContentArrayForm(t *testing.T) {
	prov := &recordingProvider{reply: "ok"}
	srv := newTestServer(t, prov)

	// Some clients send content as an array of typed parts.
	body := `{"model":"forged_reviewer","messages":[{"role":"user","content":[{"type":"text","text":"part one "},{"type":"text","text":"part two"}]}]}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	last := prov.lastReq.Messages[len(prov.lastReq.Messages)-1]
	if last.Text() != "part one part two" {
		t.Errorf("array content not joined: %q", last.Text())
	}
}

func TestChatCompletionsStreaming(t *testing.T) {
	srv := newTestServer(t, &recordingProvider{reply: "streamed answer"})

	body := `{"model":"forged_reviewer","messages":[{"role":"user","content":"go"}],"stream":true}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type = %q, want text/event-stream", ct)
	}

	out := rec.Body.String()
	if !strings.Contains(out, `"object":"chat.completion.chunk"`) {
		t.Errorf("missing chunk object in stream:\n%s", out)
	}
	if !strings.Contains(out, `"content":"streamed answer"`) {
		t.Errorf("missing content delta in stream:\n%s", out)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) {
		t.Errorf("missing finish_reason in stream:\n%s", out)
	}
	if !strings.HasSuffix(strings.TrimSpace(out), "data: [DONE]") {
		t.Errorf("stream did not end with [DONE]:\n%s", out)
	}
}
