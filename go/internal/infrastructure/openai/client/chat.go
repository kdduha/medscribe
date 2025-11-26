package client

import (
	"context"
	"log"
	"strings"
	"sync"
	"time"

	"automatic_text_processing/lab_2/internal/domain/corpus"
	"automatic_text_processing/lab_2/internal/domain/report"

	"github.com/bytedance/sonic"
	"github.com/openai/openai-go/v2"
)

const WorkersNumber int = 10

func GetChatCompletion(client *openai.Client, reports []*report.Report) ([]*corpus.Corpus, error) {
	if len(reports) == 0 {
		return nil, nil
	}

	n := len(reports)
	workers := WorkersNumber
	if workers > n {
		workers = n
	}

	errCh := make(chan error, n)
	var wg sync.WaitGroup
	var mu sync.Mutex
	results := make([]*corpus.Corpus, 0, n)

	div := n / workers
	rem := n % workers

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		start := i*div + min(i, rem)
		end := start + div
		if i < rem {
			end++
		}

		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				r := reports[j]
				chatCompletion, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage(`Ты — медицинский помощник. Тебе даны отчёты МРТ и КТ. Твоя задача — автоматически выделить в отчёте два элементы и вывести их в формате JSON:
Короткая находка — краткая формулировка (1–2 строки) главного, что видит врач, например: «Очаг 3×4 см в правой лобной доле».
Заключение — полный текст заключения из протокола, без изменений.
Формат вывода — plaintext, без дополнительного форматирования, без кода и без markdown JSON-объект с двумя полями:
{"finding": "краткая находка здесь","conclusion": "текст заключения здесь"}
Обрати внимание на требования:
Короткая находка должна быть информативной и краткой.
Заключение берётся точно из отчёта.
Если несколько находок, выбери главную для краткой находки.
Если находок нет, в поле finding укажи "нет находок".
Пример правильного ответа:
{
  "finding": "Очаг 3×4 см в правой лобной доле",
  "conclusion": "В правой лобной доле определяется очаг 3×4 см с признаками отёка. Рекомендуется консультация невролога."
}
Если в отчёте нет заключения, в поле conclusion укажи "нет заключения".
Обработай этот отчёт:
` + r.Content()),
					},
					Model: openai.ChatModelGPT4oMini,
				})
				if err != nil {
					log.Printf("Chat request failed for %s: %v", r.Filename(), err)
					c := corpus.NewCorpus("", r.Filename(), r.FolderPath(), "")
					mu.Lock()
					results = append(results, c)
					mu.Unlock()
					continue
				}

				var dto struct {
					Finding    string `json:"finding"`
					Conclusion string `json:"conclusion"`
				}

				if err := sonic.Unmarshal([]byte(chatCompletion.Choices[0].Message.Content), &dto); err != nil {
					log.Printf("Failed to unmarshal chat completion for %s: %v", r.Filename(), err)
					log.Printf("Raw response: %s", chatCompletion.Choices[0].Message.Content)
					dto.Finding = ""
					dto.Conclusion = ""
				}

				c := corpus.NewCorpus(dto.Finding, r.Filename(), r.FolderPath(), dto.Conclusion)

				mu.Lock()
				results = append(results, c)
				mu.Unlock()

				time.Sleep(30 * time.Second)
			}
		}(start, end)
	}

	wg.Wait()
	close(errCh)

	var errMsgs []string
	for e := range errCh {
		errMsgs = append(errMsgs, e.Error())
	}

	log.Printf("Processed %d out of %d reports for this batch", len(results), n)
	if len(errMsgs) > 0 {
		log.Printf("Encountered %d errors during processing: %s", len(errMsgs), strings.Join(errMsgs, "; "))
	}

	return results, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
