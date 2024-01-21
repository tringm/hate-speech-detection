download-phi2-model:
	@echo "Downloading TheBloke phi-2.Q6_K.gguf"
	@curl -L "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q6_K.gguf" --create-dirs -o models/phi-2.Q6_K.gguf


build-hatexplain-eval-data: URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/01d742279dac941981f53806154481c0e15ee686/Data/dataset.json"
build-hatexplain-eval-data: OUTPUT_FILE = "tests/data/hatexplain.ndjson"
build-hatexplain-eval-data:
	@curl -X GET $(URL) | jq -c 'to_entries | map(.value) | .[]' > $(OUTPUT_FILE)
	@gzip $(OUTPUT_FILE)
	@python -m tests.data.hatexplain
