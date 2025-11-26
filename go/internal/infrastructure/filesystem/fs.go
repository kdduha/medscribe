package filesystem

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"automatic_text_processing/lab_2/internal/domain/corpus"
	"automatic_text_processing/lab_2/internal/domain/report"
)

type FolderReports struct {
	FolderPath string
	Reports    []*report.Report
}

func FindReportsByFolder(dir string) ([]*FolderReports, int, error) {
	folderReportsMap := make(map[string][]*report.Report)
	totalReports := 0

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(d.Name(), ".txt") {
			content, err := os.ReadFile(path)
			if err != nil {
				log.Printf("Warning: Failed to read file %s: %v", path, err)
				return nil
			}
			if len(content) == 0 {
				log.Printf("Warning: Skipping empty file %s", path)
				return nil
			}

			report := report.NewReport(path, string(content))
			folderPath := report.FolderPath()

			folderReportsMap[folderPath] = append(folderReportsMap[folderPath], report)
			totalReports++
		}
		return nil
	})
	if err != nil {
		return nil, 0, err
	}

	var folderReports []*FolderReports
	for folderPath, reports := range folderReportsMap {
		folderReports = append(folderReports, &FolderReports{
			FolderPath: folderPath,
			Reports:    reports,
		})
	}

	return folderReports, totalReports, nil
}

func SaveCorporaByFolder(corporaByFolder map[string][]*corpus.Corpus) error {
	var saveErrors []string
	successfulSaves := 0

	for folderPath, corpora := range corporaByFolder {
		if len(corpora) == 0 {
			continue
		}

		outputPath := filepath.Join(folderPath, "processed_results.csv")

		if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
			saveErrors = append(saveErrors, fmt.Sprintf("failed to create directory %s: %v", filepath.Dir(outputPath), err))
			continue
		}

		f, err := os.Create(outputPath)
		if err != nil {
			saveErrors = append(saveErrors, fmt.Sprintf("failed to create output file %s: %v", outputPath, err))
			continue
		}

		writer := csv.NewWriter(f)

		header := []string{"finding", "file_name", "result"}
		if err := writer.Write(header); err != nil {
			f.Close()
			saveErrors = append(saveErrors, fmt.Sprintf("failed to write header to csv %s: %v", outputPath, err))
			continue
		}

		recordErrors := 0
		for _, c := range corpora {
			filename := filepath.Base(c.Filename())
			record := []string{c.Finding(), filename, c.Result()}
			if err := writer.Write(record); err != nil {
				recordErrors++
				log.Printf("Failed to write record %s: %v", filename, err)
				continue
			}
		}

		writer.Flush()
		f.Close()

		if err := writer.Error(); err != nil {
			saveErrors = append(saveErrors, fmt.Sprintf("error writing csv %s: %v", outputPath, err))
			continue
		}

		successfulSaves++
		if recordErrors > 0 {
			fmt.Printf("Saved %d results to %s (with %d record errors)\n", len(corpora)-recordErrors, outputPath, recordErrors)
		} else {
			fmt.Printf("Saved %d results to %s\n", len(corpora), outputPath)
		}
	}

	if len(saveErrors) > 0 {
		log.Printf("Some save operations failed: %v", strings.Join(saveErrors, "; "))
		if successfulSaves == 0 {
			return fmt.Errorf("all save operations failed: %s", strings.Join(saveErrors, "; "))
		}
		return nil
	}

	return nil
}

func SaveCorpora(corpora []*corpus.Corpus) error {
	f, err := os.Create("output.csv")
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	defer writer.Flush()

	header := []string{"finding", "file_name", "result"}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write header to csv: %w", err)
	}

	for _, c := range corpora {
		filename := filepath.Base(c.Filename())
		record := []string{c.Finding(), filename, c.Result()}
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("failed to write record to csv: %w", err)
		}
	}

	return nil
}
