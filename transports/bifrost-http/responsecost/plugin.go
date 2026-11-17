// Package responsecost puts each call's cost in its response as usage.cost: the same
// figure Bifrost logs and budgets with (full pricing resolution: cache rates,
// long-context tiers, service tiers, pricing overrides), so a client can bill from the
// response instead of keeping its own price list. A cost the provider reported itself
// (OpenRouter) is left as is.
package responsecost

import "github.com/maximhq/bifrost/core/schemas"

// PluginName is the name this plugin is configured and registered under.
const PluginName = "response-cost"

// Plugin fills usage.cost on chat, Responses (the final streamed event included) and
// embeddings replies.
type Plugin struct{}

// Init returns the plugin. It needs no configuration: the cost comes from the model
// catalog the request context carries.
func Init() (*Plugin, error) { return &Plugin{}, nil }

func (p *Plugin) GetName() string { return PluginName }

func (p *Plugin) Cleanup() error { return nil }

func (p *Plugin) PreRequestHook(_ *schemas.BifrostContext, _ *schemas.BifrostRequest) error {
	return nil
}

func (p *Plugin) PreLLMHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}

// PostLLMHook sets usage.cost on a reply that carries usage but no cost. It runs
// synchronously, as CalculateCostBreakdown requires (it reads the request's pricing
// scopes off the context).
func (p *Plugin) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if ctx == nil || resp == nil {
		return resp, bifrostErr, nil
	}
	if cost := costSlot(resp); cost != nil && *cost == nil {
		*cost = ctx.CalculateCostBreakdown(resp)
	}
	return resp, bifrostErr, nil
}

// costSlot is the usage.cost field of resp's reply, or nil when the reply carries no
// usage (a streamed chunk before the last, an unpriced request type).
func costSlot(resp *schemas.BifrostResponse) **schemas.BifrostCost {
	switch {
	case resp.ChatResponse != nil && resp.ChatResponse.Usage != nil:
		return &resp.ChatResponse.Usage.Cost
	case resp.ResponsesResponse != nil && resp.ResponsesResponse.Usage != nil:
		return &resp.ResponsesResponse.Usage.Cost
	case resp.ResponsesStreamResponse != nil && resp.ResponsesStreamResponse.Response != nil &&
		resp.ResponsesStreamResponse.Response.Usage != nil:
		return &resp.ResponsesStreamResponse.Response.Usage.Cost
	case resp.EmbeddingResponse != nil && resp.EmbeddingResponse.Usage != nil:
		return &resp.EmbeddingResponse.Usage.Cost
	}
	return nil
}
