package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
)

func TestResponsesNonStreaming(t *testing.T) {
	prov := &recordingProvider{reply: "a grounded review"}
	srv := newTestServer(t, prov)

	// Responses-style input: array of message items with input_text parts.
	body := `{"model":"fired_reviewer","instructions":"codex system prompt",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"review this repo"}]}]}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var resp responsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Object != "response" || resp.Status != "completed" {
		t.Errorf("object/status = %q/%q", resp.Object, resp.Status)
	}
	if len(resp.Output) != 1 || len(resp.Output[0].Content) != 1 {
		t.Fatalf("unexpected output: %+v", resp.Output)
	}
	if got := resp.Output[0].Content[0].Text; got != "a grounded review" {
		t.Errorf("output text = %q", got)
	}
	if resp.Output[0].Content[0].Type != "output_text" {
		t.Errorf("part type = %q, want output_text", resp.Output[0].Content[0].Type)
	}

	// The input_text part must have reached the provider.
	last := prov.lastReq.Messages[len(prov.lastReq.Messages)-1]
	if last.Text() != "review this repo" {
		t.Errorf("user message not forwarded: %q", last.Text())
	}
	if prov.lastReq.SystemPrompt != "SCAFFOLD" {
		t.Errorf("scaffold not forwarded: %q", prov.lastReq.SystemPrompt)
	}
}

func TestResponsesStringInput(t *testing.T) {
	prov := &recordingProvider{reply: "ok"}
	srv := newTestServer(t, prov)

	body := `{"model":"fired_reviewer","input":"just a string"}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	last := prov.lastReq.Messages[len(prov.lastReq.Messages)-1]
	if last.Text() != "just a string" {
		t.Errorf("string input not forwarded: %q", last.Text())
	}
}

func TestResponsesStreaming(t *testing.T) {
	srv := newTestServer(t, &recordingProvider{reply: "streamed answer"})

	body := `{"model":"fired_reviewer","input":"go","stream":true}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type = %q", ct)
	}

	out := rec.Body.String()
	for _, want := range []string{
		"event: response.created",
		"event: response.output_text.delta",
		`"delta":"streamed answer"`,
		"event: response.completed",
		`"status":"completed"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stream missing %q in:\n%s", want, out)
		}
	}

	// Every event should carry an incrementing sequence_number.
	if !strings.Contains(out, `"sequence_number":0`) || !strings.Contains(out, `"sequence_number":1`) {
		t.Errorf("missing sequence numbers in:\n%s", out)
	}
}

// TestResponsesZstdEncodedBody reproduces what Codex actually sends: a
// zstd-compressed request body with Content-Encoding: zstd.
func TestResponsesZstdEncodedBody(t *testing.T) {
	prov := &recordingProvider{reply: "decompressed ok"}
	srv := newTestServer(t, prov)

	payload := `{"model":"fired_reviewer","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello zstd"}]}]}`
	enc, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatalf("zstd writer: %v", err)
	}
	compressed := enc.EncodeAll([]byte(payload), nil)
	enc.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(compressed))
	req.Header.Set("Content-Encoding", "zstd")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	last := prov.lastReq.Messages[len(prov.lastReq.Messages)-1]
	if last.Text() != "hello zstd" {
		t.Errorf("zstd body not decoded: %q", last.Text())
	}
}

// TestResponsesZstdSniffedNoHeader covers a compressed body with no
// Content-Encoding header — decoded via magic-byte sniffing.
func TestResponsesZstdSniffedNoHeader(t *testing.T) {
	prov := &recordingProvider{reply: "ok"}
	srv := newTestServer(t, prov)

	payload := `{"model":"fired_reviewer","input":"sniffed"}`
	enc, _ := zstd.NewWriter(nil)
	compressed := enc.EncodeAll([]byte(payload), nil)
	enc.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(compressed))
	// deliberately no Content-Encoding header
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	last := prov.lastReq.Messages[len(prov.lastReq.Messages)-1]
	if last.Text() != "sniffed" {
		t.Errorf("sniffed zstd body not decoded: %q", last.Text())
	}
}

// TestResponsesDefaultAgentFallback reproduces the GUI case: a host that sends
// its own model id (e.g. "gpt-5.5") should fall back to the default agent
// rather than 404.
func TestResponsesDefaultAgentFallback(t *testing.T) {
	prov := &recordingProvider{reply: "ok"}
	srv := newTestServerWithDefault(t, prov, "fired_reviewer")

	body := `{"model":"gpt-5.5","input":"hi from the GUI"}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (fallback should apply) (%s)", rec.Code, rec.Body.String())
	}
	// The scaffold of the default agent must still be applied.
	if prov.lastReq.SystemPrompt != "SCAFFOLD" {
		t.Errorf("default agent scaffold not applied: %q", prov.lastReq.SystemPrompt)
	}
}

func TestResponsesUnknownModelNoDefault(t *testing.T) {
	srv := newTestServer(t, &recordingProvider{reply: "ok"}) // no default

	body := `{"model":"gpt-5.5","input":"hi"}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (strict, no default)", rec.Code)
	}
}

func TestResponsesUnknownModel(t *testing.T) {
	srv := newTestServer(t, &recordingProvider{reply: "ok"})

	body := `{"model":"nope","input":"hi"}`
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
}
