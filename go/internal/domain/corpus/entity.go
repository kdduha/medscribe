package corpus

type Corpus struct {
	finding    string
	filename   string
	folderPath string
	result     string
}

func NewCorpus(finding, filename, folderPath, result string) *Corpus {
	return &Corpus{
		finding:    finding,
		filename:   filename,
		folderPath: folderPath,
		result:     result,
	}
}

func (c *Corpus) Finding() string {
	return c.finding
}

func (c *Corpus) Filename() string {
	return c.filename
}

func (c *Corpus) FolderPath() string {
	return c.folderPath
}

func (c *Corpus) Result() string {
	return c.result
}
