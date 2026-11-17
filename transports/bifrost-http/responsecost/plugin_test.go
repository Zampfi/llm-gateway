package responsecost

import (
	"context"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// fakeCatalog prices every reply at a fixed breakdown, standing in for the model catalog
// Bifrost puts on the request context.
type fakeCatalog struct{ cost *schemas.BifrostCost }

func (f fakeCatalog) GetModelInfo(schemas.ModelProvider, string) *schemas.Model { return nil }

func (f fakeCatalog) CalculateRequestCost(*schemas.BifrostContext, *schemas.BifrostResponse) float64 {
	return f.cost.TotalCost
}

func (f fakeCatalog) CalculateRequestCostBreakdown(*schemas.BifrostContext, *schemas.BifrostResponse) *schemas.BifrostCost {
	c := *f.cost
	return &c
}

func ctxWith(catalog schemas.ModelInfoProvider) *schemas.BifrostContext {
	if catalog == nil {
		return schemas.NewBifrostContext(context.Background(), time.Now().Add(time.Minute))
	}
	return schemas.NewBifrostContextWithValue(
		context.Background(), time.Now().Add(time.Minute), schemas.BifrostContextKeyModelCatalog, catalog,
	)
}

func TestPostLLMHookPutsTheCallsCostInTheReply(t *testing.T) {
	priced := &schemas.BifrostCost{InputCost: 0.002, OutputCost: 0.004, TotalCost: 0.006}
	reported := &schemas.BifrostCost{TotalCost: 0.5}
	usage := func() *schemas.BifrostLLMUsage {
		return &schemas.BifrostLLMUsage{PromptTokens: 10, CompletionTokens: 5}
	}
	respUsage := func() *schemas.ResponsesResponseUsage {
		return &schemas.ResponsesResponseUsage{InputTokens: 10, OutputTokens: 5}
	}

	cases := []struct {
		name    string
		catalog schemas.ModelInfoProvider
		resp    *schemas.BifrostResponse
		cost    func(*schemas.BifrostResponse) *schemas.BifrostCost
		want    *schemas.BifrostCost
	}{
		{
			name:    "chat reply",
			catalog: fakeCatalog{priced},
			resp:    &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{Usage: usage()}},
			cost:    func(r *schemas.BifrostResponse) *schemas.BifrostCost { return r.ChatResponse.Usage.Cost },
			want:    priced,
		},
		{
			name:    "responses reply",
			catalog: fakeCatalog{priced},
			resp:    &schemas.BifrostResponse{ResponsesResponse: &schemas.BifrostResponsesResponse{Usage: respUsage()}},
			cost:    func(r *schemas.BifrostResponse) *schemas.BifrostCost { return r.ResponsesResponse.Usage.Cost },
			want:    priced,
		},
		{
			name:    "streamed responses completed event",
			catalog: fakeCatalog{priced},
			resp: &schemas.BifrostResponse{ResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
				Response: &schemas.BifrostResponsesResponse{Usage: respUsage()},
			}},
			cost: func(r *schemas.BifrostResponse) *schemas.BifrostCost {
				return r.ResponsesStreamResponse.Response.Usage.Cost
			},
			want: priced,
		},
		{
			name:    "embeddings reply",
			catalog: fakeCatalog{priced},
			resp:    &schemas.BifrostResponse{EmbeddingResponse: &schemas.BifrostEmbeddingResponse{Usage: usage()}},
			cost:    func(r *schemas.BifrostResponse) *schemas.BifrostCost { return r.EmbeddingResponse.Usage.Cost },
			want:    priced,
		},
		{
			name:    "a cost the provider reported is kept",
			catalog: fakeCatalog{priced},
			resp: &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{
				Usage: &schemas.BifrostLLMUsage{PromptTokens: 10, Cost: reported},
			}},
			cost: func(r *schemas.BifrostResponse) *schemas.BifrostCost { return r.ChatResponse.Usage.Cost },
			want: reported,
		},
		{
			name:    "no catalog on the context leaves the cost empty",
			catalog: nil,
			resp:    &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{Usage: usage()}},
			cost:    func(r *schemas.BifrostResponse) *schemas.BifrostCost { return r.ChatResponse.Usage.Cost },
			want:    nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, _ := Init()
			out, _, err := p.PostLLMHook(ctxWith(tc.catalog), tc.resp, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got := tc.cost(out)
			if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
				t.Fatalf("usage.cost = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestPostLLMHookLeavesAReplyWithoutUsageAlone(t *testing.T) {
	p, _ := Init()
	chunk := &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{}}
	out, _, err := p.PostLLMHook(ctxWith(fakeCatalog{&schemas.BifrostCost{TotalCost: 1}}), chunk, nil)
	if err != nil || out.ChatResponse.Usage != nil {
		t.Fatalf("a chunk without usage must stay without usage, got %+v (err %v)", out.ChatResponse.Usage, err)
	}
}
