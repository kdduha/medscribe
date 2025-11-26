package config

import (
	"fmt"

	"github.com/spf13/viper"
)

type Config struct {
	AITunnelBaseUrl string `env:"AI_TUNNEL_BASE_URL"`
	AITunnelAPIKey  string `env:"AI_TUNNEL_API_KEY,required"`
	InputDir        string `env:"INPUT_DIR"`
}

func NewConfig() (*Config, error) {
	viper.SetConfigFile(".env")
	viper.SetConfigType("env")
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, err
		}
	}

	viper.AutomaticEnv()

	viper.SetDefault("AI_TUNNEL_BASE_URL", "https://api.openai.com/v1")

	cfg := &Config{
		AITunnelBaseUrl: viper.GetString("AI_TUNNEL_BASE_URL"),
		AITunnelAPIKey:  viper.GetString("AI_TUNNEL_API_KEY"),
		InputDir:        viper.GetString("INPUT_DIR"),
	}

	if cfg.AITunnelAPIKey == "" {
		return nil, fmt.Errorf("AI_TUNNEL_API_KEY is required")
	}

	return cfg, nil
}
