package client

import (
	"fmt"
	"net/url"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

func NewClient(apiKey, baseUrl string) (*openai.Client, error) {
	if apiKey == "" || baseUrl == "" {
		return nil, fmt.Errorf("apiKey and baseUrl must be provided")
	}

	if _, err := url.Parse(baseUrl); err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseUrl),
	)

	return &client, nil
}
