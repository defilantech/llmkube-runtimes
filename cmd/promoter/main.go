package main

import (
	"context"
	"fmt"
	"os"

	"github.com/defilantech/llmkube-runtimes/internal/promoter"
	"github.com/spf13/cobra"
)

func main() {
	var cfg promoter.Config
	root := &cobra.Command{Use: "promoter", SilenceUsage: true}
	runOnce := &cobra.Command{
		Use:   "run-once",
		Short: "Process new candidate images once and exit",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return promoter.RunOnce(cmd.Context(), cfg)
		},
	}
	f := runOnce.Flags()
	f.StringVar(&cfg.Repo, "repo", "ghcr.io/defilantech/llmkube-llama-vulkan", "image repository")
	f.StringVar(&cfg.AttestRepo, "attest-repo", "defilantech/llmkube-runtimes", "repo that built+attested the image")
	f.StringVar(&cfg.StatePath, "state", os.ExpandEnv("$HOME/.local/state/llmkube-promoter/processed.json"), "state file")
	f.StringVar(&cfg.Namespace, "namespace", "llmkube-promoter", "smoke Job namespace")
	f.IntVar(&cfg.RenderGID, "render-gid", 0, "render group GID for GPU device access")
	f.Float64Var(&cfg.MinDecodeTokS, "min-decode-toks", 40, "minimum decode tokens/sec to pass smoke")
	root.AddCommand(runOnce)
	if err := root.ExecuteContext(context.Background()); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
