# Hate Speech Detection App

An app that uses LLM to flag hate speech text and explain the reasoning.

## Background

Identifying hate speech can be challenging. One of the main reasons is that there's no universal definition and
hate speech varies on social, cultural, and historical contexts.

Furthermore, rather than just identifying whether a text is a hate speech or not, the system should be able to
explain the reasoning behind the decision. Providing the reasoning can provide the following benefits:
- Transparency: Users should understand why a piece of content was flagged as hate speech.
This transparency builds trust in the system and helps users comprehend the criteria used for identifying hate speech.
Based on this context, the user can flag incorrect cases and provide valuable feedback.
- User Education: The system can help the users to be aware of the boundaries and guidelines and behave accordingly.
- Continuous improvement: The reasoning combined with feedback from the users can be used to improve the system in further iteration.

This app can be used an add-on to flag potential hate speech in communication.

Furthermore, by adding the context to the prompt, the app can detect text that violates local guidelines.

## Approach

Main components:
- [Microsoft Phi 2 LLM](https://huggingface.co/TheBloke/phi-2-GGUF) loaded with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) to idenity hate speech and generate explanation
  - The model is a "small language model" promised with reasonable performance
  - In combination with Llama CPP, the model can be run using only CPU and requires 4.79 GB of RAM. The model can be attached to a Docker image, and served as a cloud function, for example.
  - The model is available for future fine-tuning, and guarantee that the data stays in the system.
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API.
- [PostgreSQL DB](https://www.postgresql.org/) to track LLM run and API calls.
  - "llm_run" table stores LLM prompt run information such as the llm model in used, model configs, prompt, run configs, etc.
  - "detect_hate_speech_result" table stores hate speech detection requests


## Experimentation

A modified version of [HateXplain](https://github.com/hate-alert/HateXplain) dataset was used to benchmark the system.

In each case of the dataset, there are 3 annotators annotating whether a text is Normal | Offensive | Hate, the target groups, and the words in the text that imply hate.

The test set only uses case which all 3 annotators unanimously agree on the text label, and group the label into Hate (Offensive and Hate) and Non-Hate (Normal).

### LLM models
- Besides Phi-2, [Llama 2](https://ai.meta.com/llama/) was also used. Llama 2 can also be used with Llama-CPP.
- Llama 2 provides a nice feature of adding a "System prompt". For example
  ```
  <INST>
  <<SYS>>
  You are a helpful and diligent moderator that can accurately flag hate speech.
  You are able to differentiating between strongly opinionated language and genuinely offensive content.
  <</SYS>>
  Determine if the following text is a hate speech.
  {input_text}
  </INST>
  ```

### Prompting strategy
Prompting strategy used:
- Zero-shot:
  ```
  Determine if the following text is a hate speech.
  Explain in the reasoning and identify the target of hate if the text is a hate speech.

  The text is enclosed in backticks:
  `{input_text}`
  ```

- Few-Shot Prompting: Providing a definition of hate speech as context (e.g.: [UN definition](https://www.un.org/en/hate-speech/understanding-hate-speech/what-is-hate-speech))
  ```
  Hate speech can be defined as any kind of communication in speech, writing or behaviour, that attacks or uses
   pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other
   words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.”

  Example:
  Input Text: <example_non_hate>
  Result: {"is_hate_speech":false,"target_of_hate":[],"reasoning":""}

  Input Text:  <example_hate>
  Result: {"is_hate_speech":false,"target_of_hate":[],"reasoning":""}
  ```
  While few-shot prompting can provide some improvements, it increases the context length and leads to slower generation.
- Chain-of-thought Prompting (currently in used): Ask the LLm to returns the reasoning steps. For example, zero-shot with explanation:
  ```
  Determine if the following text is a hate speech.
  Explain step-by-step the reasoning and identify the target of hate if the text is a hate speech.

  The text is enclosed in backticks: ...
  ```
  **NOTE**: GPT 3.5 performs much better with this strategy, even better with few-shot with COT. The model does return good step-by-step reasoning in contrast to Llama 2 and Phi which doesn't really have COT.


## Key Learnings
- Llama CPP supports [GBNF grammar](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) which allow forcing prompt output in a certain format (e.g.: JSON).
- [SQLModel](https://sqlmodel.tiangolo.com/) that can be used in conjunction with FastAPI, SQLAlchemy, and Alembic
- The solution uses a custom DB to track LLM runs. There must be an existing solution that can do this better.
For example, MLFlow offers [LLM tracking capability](https://mlflow.org/docs/latest/llms/llm-tracking/index.html). However, based on a quick look, it's bounded to their
implementation of using OpenAI API Endpoint.
- While Phi 2 achieves reasonable performance, GPT 3.5 performs much better in both hate speech detection and giving explanation.


## Implementation
### Run the app

First, download the Phi-2 model:

```shell
make download-phi2-model
```

Start the stack with [docker compose](https://docs.docker.com/compose/:

```shell
docker-compose build
docker-compose up app
```

The API is available at `http://localhost:8080`:
- Check out the [API docs](http://localhost:8080/docs)
- Example:
  ```shell
  curl --location 'http://172.17.0.1:8080/detect_hate_speech' \
  --header 'Content-Type: application/json' \
  --data '{"text": "We have enough problems in the world. Stop hating on each other"}'
  ```
  Output:
  ```json
  {
    "is_hate_speech": false,
    "target_of_hate": [],
    "reasoning": "",
    "text": "We have enough problems in the world. Stop hating on each other",
    "llm_run_id": "e1feb7e2-1855-4a7a-80ea-1afe1bc4081e",
    "uuid": "4e954eb6-7c9f-4ca7-9f5b-321ab84f3ad0"
  }
  ```


### Pre-commit

The project uses pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `ruff`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

### Testing

To run unittests:

```shell
pytest
```

To run integration tests:

```shell
docker compose run integration-tests
```

To run long-running evaluation tests (e.g.: LLM evaluation):

```shell
pytest --run-eval -k evaluation
```

To enable comparison mode that compares the generated output after test finishes:

```shell
pytest --output-diff --run-eval -k evaluation
```
