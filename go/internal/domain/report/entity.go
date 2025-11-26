package report

import (
	"path/filepath"
)

type Report struct {
	filename string
	filepath string
	content  string
}

func NewReport(filepath string, content string) *Report {
	filename := filepath
	return &Report{
		filename: filename,
		filepath: filepath,
		content:  content,
	}
}

func (r Report) Filename() string {
	return r.filename
}

func (r Report) Filepath() string {
	return r.filepath
}

func (r Report) FolderPath() string {
	return filepath.Dir(r.filepath)
}

func (r Report) Content() string {
	return r.content
}
