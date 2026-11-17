package anthropic

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// A Claude reply that thinks between tool calls must come back with each thinking block
// where Claude put it. Replaying a reply whose thinking was moved to the front makes the
// next turn fail with "thinking blocks in the latest assistant message cannot be
// modified" (issue #9156). The request side already keeps the order (issue #7768).
func TestToBifrostResponsesResponseKeepsClaudeThinkingOrder(t *testing.T) {
	str := func(s string) *string { return &s }
	th := func(text, sig string) AnthropicContentBlock {
		return AnthropicContentBlock{Type: AnthropicContentBlockTypeThinking, Thinking: str(text), Signature: str(sig)}
	}
	txt := func(text string) AnthropicContentBlock {
		return AnthropicContentBlock{Type: AnthropicContentBlockTypeText, Text: str(text)}
	}
	tool := func(id string) AnthropicContentBlock {
		return AnthropicContentBlock{Type: AnthropicContentBlockTypeToolUse, ID: str(id), Name: str("get_invoice"), Input: []byte(`{}`)}
	}
	resp := &AnthropicMessageResponse{
		ID:      "msg_1",
		Model:   "claude-sonnet-4-5",
		Content: []AnthropicContentBlock{th("a", "sig0"), txt("x"), tool("t0"), th("b", "sig1"), txt("y"), tool("t1")},
		Usage:   &AnthropicUsage{},
	}

	out := resp.ToBifrostResponsesResponse(&schemas.BifrostContext{})

	var got []string
	for _, m := range out.Output {
		switch {
		case m.Type != nil && *m.Type == schemas.ResponsesMessageTypeReasoning:
			got = append(got, "reasoning:"+fmt.Sprint(len(m.Content.ContentBlocks)))
		case m.Type != nil && *m.Type == schemas.ResponsesMessageTypeFunctionCall:
			got = append(got, "call:"+*m.ResponsesToolMessage.CallID)
		default:
			got = append(got, "message")
		}
	}
	want := []string{"reasoning:1", "message", "call:t0", "reasoning:1", "message", "call:t1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("thinking order not preserved in the converted reply:\n got  %v\n want %v", got, want)
	}
}
