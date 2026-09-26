// Command kiln-gateway serves forged agents behind a local OpenAI-compatible
// endpoint. It is the runnable proof of the serve path: it wires demo agents
// (defined in Go) to the OpenAI gateway so you can point any OpenAI client at
// it.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...            # key for the upstream provider
//	go run ./cmd/kiln-gateway --addr :8787
//
//	export OPENAI_BASE_URL=http://localhost:8787/v1
//	export OPENAI_API_KEY=kiln-local       # the gateway ignores this
//	# now start Codex CLI / Grok Build, or just curl /v1/chat/completions
//
// The two demo agents share one provider and model and differ only in their
// scaffold — the north-star demo: same model, same task, different package.
//
// This command owns process lifecycle: it starts the Gateway, waits for an
// interrupt/terminate signal, then asks the Gateway to shut down gracefully.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/katasec/kiln"
	"github.com/katasec/kiln-gateway/gateway"
	"github.com/katasec/kiln/provider/openai"
)

const forgedReviewerScaffold = `You are a repository reviewer operating under a mission scaffold.

Operating rules:
- Start from a small orientation layer; do not assume the whole repo.
- Ground every finding in a concrete file, command, or repo fact — no generic advice.
- Identify exactly one concrete, high-value next improvement.
- Recommend the verification (build/test/lint) that proves the change.
- Keep output structured: Findings, Next Improvement, Verification.`

const vanillaScaffold = "You are a helpful coding assistant. Review the repository."

// shutdownTimeout bounds how long Stop waits for in-flight requests to drain.
const shutdownTimeout = 10 * time.Second

func main() {
	addr := flag.String("addr", ":8787", "address to listen on")
	model := flag.String("model", string(openai.ModelGPT54Nano), "upstream model id")
	baseURL := flag.String("base-url", "", "override upstream OpenAI base URL (e.g. for xAI)")
	defaultAgent := flag.String("default-agent", "forged_reviewer",
		"agent to use when a client requests an unknown model id (host GUIs send their own model names); empty for strict 404")
	flag.Parse()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required (the upstream provider key)")
	}

	var opts []openai.Option
	if *baseURL != "" {
		opts = append(opts, openai.WithBaseURL(*baseURL))
	}
	provider := openai.New(apiKey, openai.Model(*model), opts...)

	agents := map[string]*kiln.Agent{
		"vanilla_reviewer": mustAgent(provider, vanillaScaffold),
		"forged_reviewer":  mustAgent(provider, forgedReviewerScaffold),
	}

	gw := gateway.New(gateway.Config{
		Addr:         *addr,
		Agents:       agents,
		DefaultAgent: *defaultAgent,
	})

	log.Printf("kiln-gateway serving %d agents on %s (upstream model %s, default agent %q)", len(agents), *addr, *model, *defaultAgent)
	log.Printf("point your client at: export OPENAI_BASE_URL=http://localhost%s/v1", *addr)

	// main owns OS signal handling; the gateway owns Start/Stop.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- gw.Start(ctx) }()

	select {
	case err := <-errCh:
		if err != nil {
			log.Fatalf("kiln-gateway: %v", err)
		}
	case <-ctx.Done():
		stop() // restore default signal handling so a second signal force-quits
		log.Printf("kiln-gateway shutting down…")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		if err := gw.Stop(shutdownCtx); err != nil {
			log.Fatalf("kiln-gateway shutdown: %v", err)
		}
		log.Printf("kiln-gateway stopped")
	}
}

func mustAgent(provider kiln.Provider, scaffold string) *kiln.Agent {
	agent, err := kiln.NewAgent(kiln.Config{
		Provider:      provider,
		SystemPrompt:  scaffold,
		DisableMemory: true, // OpenAI clients are stateless; they resend history
	})
	if err != nil {
		log.Fatal(err)
	}
	return agent
}
