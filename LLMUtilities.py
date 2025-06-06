# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/levitation-opensource/bioblue

import os
import time

import tenacity
import tiktoken

import traceback
import httpcore
import httpx
import json
import json_tricks

import openai

from Utilities import Timer, wait_for_enter

import configparser
import ast


config_path = r"config.ini" 
config = configparser.ConfigParser()
config.read_file(open(config_path))

model_name = ast.literal_eval(config.get('Model params', 'name'))

# Initialize the appropriate client based on the model name
if model_name.lower().startswith('claude'):
    from anthropic import Anthropic  
    claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("Initialized Claude client")
elif model_name.lower().startswith('gpt'):
    from openai import OpenAI  
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Initialized OpenAI client")
elif model_name.lower().startswith('local'):
    from openai import OpenAI
    # base_url : https://github.com/openai/openai-python/issues/1051
    # do not set OPENAI_BASE_URL env variable since that would override 
    # the normal GPT model usage config as well
    base_url = os.getenv("CUSTOM_OPENAI_BASE_URL", "http://localhost:1234")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
    print("Initialized Local client")
else:
    print(f"Unsupported model: {model_name}")


## https://platform.openai.com/docs/guides/rate-limits/error-mitigation
# TODO: config parameter for max attempt number
@tenacity.retry(
  wait=tenacity.wait_random_exponential(min=1, max=60),
  stop=tenacity.stop_after_attempt(10),
)  # TODO: config parameters
def completion_with_backoff(
  gpt_timeout, **kwargs
):  # TODO: ensure that only HTTP 429 is handled here
  # return openai.ChatCompletion.create(**kwargs)

  attempt_number = completion_with_backoff.retry.statistics["attempt_number"]
  max_attempt_number = completion_with_backoff.retry.stop.max_attempt_number
  timeout_multiplier = 2 ** (attempt_number - 1)  # increase timeout exponentially

  try:
    timeout = gpt_timeout * timeout_multiplier

    # print(f"Sending LLM API request... Using timeout: {timeout} seconds")

    # TODO!!! support for other LLM API-s
    is_claude = model_name.startswith('claude-')
    if is_claude:
    
      # TODO!!! implement automatic rate limiting for Anthropic API

      messages = kwargs.pop('messages', [])
      system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        
      # Build the messages for Claude
      claude_messages = []
      claude_messages = [msg for msg in messages if msg['role'] != 'system']
      response = claude_client.messages.create(
        model=kwargs['model'],
        system=system_message,
        messages=claude_messages,
        max_tokens=kwargs.get('max_tokens', 1024),
        temperature=kwargs.get('temperature', 0.5)
      )
      return (response.content[0].text, response.stop_reason, response.usage.input_tokens, response.usage.output_tokens)
      
    else:

      # set openai internal max_retries to 1 so that we can log errors to console
      openai_response = openai_client.with_options(
        timeout=gpt_timeout, max_retries=1
      ).with_raw_response.chat.completions.create(**kwargs)

      # print("Done OpenAI API request.")

      openai_response = json_tricks.loads(
        openai_response.content.decode("utf-8", "ignore")
      )

      if openai_response.get("error"):
        if (
          openai_response["error"]["code"] == 502
          or openai_response["error"]["code"] == 503
        ):  # Bad gateway or Service Unavailable
          raise httpcore.NetworkError(openai_response["error"]["message"])
        else:
          raise Exception(
            str(openai_response["error"]["code"])
            + " : "
            + openai_response["error"]["message"]
          )  # TODO: use a more specific exception type

      # NB! this line may also throw an exception if the OpenAI announces that it is overloaded # TODO: do not retry for all error messages
      response_content = openai_response["choices"][0]["message"]["content"]
      finish_reason = openai_response["choices"][0]["finish_reason"]

      return (response_content, finish_reason, None, None)  # TODO: input_tokens, output_tokens

  except Exception as ex: 
    t = type(ex)  

    if t is httpcore.ReadTimeout or t is httpx.ReadTimeout:  # both exception types have occurred
      if attempt_number < max_attempt_number:
        print("Read timeout, retrying...")
      else:
        print("Read timeout, giving up")

    elif t is httpcore.NetworkError:
      if attempt_number < max_attempt_number:
        print("Network error, retrying...")
      else:
        print("Network error, giving up")

    elif t is json.decoder.JSONDecodeError:
      if attempt_number < max_attempt_number:
        print("Response format error, retrying...")
      else:
        print("Response format error, giving up")

    elif t is openai.RateLimitError:    # TODO: add support for Claude rate limit error as well    # TODO: detect when the credit limit is exceeded
      if attempt_number < max_attempt_number:
        print("Rate limit error, retrying...")
      else:
        wait_for_enter("Rate limit error. Press any key to retry")

    else:  # / if (t ishttpcore.ReadTimeout
      msg = f"{str(ex)}\n{traceback.format_exc()}"
      print(msg)

      wait_for_enter("Press any key to retry")

    # / if (t ishttpcore.ReadTimeout

    raise

  # / except Exception as ex:

# / def completion_with_backoff(gpt_timeout, **kwargs):


def get_encoding_for_model(model):

  # TODO: gpt-4.5 encoding is still unknown
  if model.startswith("gpt-4.1"): # https://github.com/openai/tiktoken/issues/395
    encoding = tiktoken.get_encoding("o200k_base")  # https://huggingface.co/datasets/openai/mrcr#how-to-run
  else:
    try:
      encoding = tiktoken.encoding_for_model(model)
    except KeyError:
      print("Warning: model not found. Using cl100k_base encoding.")
      encoding = tiktoken.get_encoding("cl100k_base")

  return encoding

# / def get_encoding_for_model(model):


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model, encoding=None):
  """Return the number of tokens used by a list of messages."""
  
  is_local = model.lower().startswith("local")
  is_claude = model.lower().startswith('claude-')
  
  if is_local:

    return 0    # TODO

  elif is_claude:

    return 0    # TODO

  else: # OpenAI

    if encoding is None:
      encoding = get_encoding_for_model(model)

    if model in {
      "gpt-3.5-turbo-0125",
      "gpt-3.5-turbo-0613",
      "gpt-3.5-turbo-16k-0613",
      "gpt-4-0314",
      "gpt-4-0613",
      "gpt-4-32k-0314",
      "gpt-4-32k-0613",
      "gpt-4o-mini-2024-07-18",
      "gpt-4o-2024-08-06",
    }:
      tokens_per_message = 3
      tokens_per_name = 1

    elif model == "gpt-3.5-turbo-0301":
      tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n         
      tokens_per_name = -1  # if there's a name, the role is omitted

    elif "gpt-3.5-turbo-16k" in model:  # roland
      # print("Warning: gpt-3.5-turbo-16k may update over time. Returning num tokens assuming gpt-3.5-turbo-16k-0613.")
      return num_tokens_from_messages(
        messages, model="gpt-3.5-turbo-16k-0613", encoding=encoding
      )

    elif "gpt-3.5-turbo" in model:
      # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
      return num_tokens_from_messages(
        messages, model="gpt-3.5-turbo-0613", encoding=encoding
      )

    elif "gpt-4-32k" in model:  # roland
      # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-32k-0613.")
      return num_tokens_from_messages(
        messages, model="gpt-4-32k-0613", encoding=encoding
      )

    elif "gpt-4o-mini" in model:
      # print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
      return num_tokens_from_messages(
        messages, model="gpt-4o-mini-2024-07-18", encoding=encoding
      )

    elif "gpt-4o" in model:
      # print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
      return num_tokens_from_messages(
        messages, model="gpt-4o-2024-08-06", encoding=encoding
      )

    elif "gpt-4" in model:
      # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
      return num_tokens_from_messages(messages, model="gpt-4-0613", encoding=encoding)

    else:
      # raise NotImplementedError(
      #  f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
      # )
      print(f"num_tokens_from_messages() is not implemented for model {model}")
      # just take some conservative assumptions here
      tokens_per_message = 4
      tokens_per_name = 1

    num_tokens = 0
    for message in messages:
      num_tokens += tokens_per_message

      for key, value in message.items():
        if key == "weight":
          continue

        num_tokens += len(encoding.encode(value))
        if key == "name":
          num_tokens += tokens_per_name

      # / for key, value in message.items():

    # / for message in messages:

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    return num_tokens

# / def num_tokens_from_messages(messages, model, encoding=None):


def get_max_tokens_for_model(model_name):
  # TODO: config
  
  is_claude = model_name.startswith('claude-')
  
  if is_claude:
       
    # Adding Claude model token limits
    claude_limits = {
      # https://aws.amazon.com/bedrock/claude/
      # TODO: check whether the listing below is complete

      'claude-opus-4-20250514': 200000,
      'claude-sonnet-4-20250514': 200000,

      'claude-3-5-sonnet-latest': 200000,
      'claude-3-5-haiku-latest': 200000,

      'claude-3-5-sonnet-20241022': 200000,
      'claude-3-5-haiku-20241022': 200000,

      'claude-3-opus-latest': 200000,
      'claude-3-sonnet-latest': 200000,
      'claude-3-haiku-latest': 200000,

      'claude-3-opus-20240229': 200000,
      'claude-3-sonnet-20240229': 200000,

      'claude-3-haiku-20240307': 200000,

      'claude-2.1': 200000,
      'claude-2.0': 100000,
      'claude-instant': 100000,
    }
    
    if model_name in claude_limits:
      max_tokens = claude_limits[model_name]
    else:
      assert False  # you probably have to add your model name to above list
      max_tokens = 100000 # 4096

  # OpenAI models # TODO: refactor to use dictionary like claude's branch uses
  elif model_name == "o1":  # https://platform.openai.com/docs/models/#o1
    max_tokens = 200000
  elif model_name == "o1-2024-12-17":  # https://platform.openai.com/docs/models/#o1
    max_tokens = 200000
  elif model_name == "o1-mini":  # https://platform.openai.com/docs/models/#o1
    max_tokens = 128000
  elif (
    model_name == "o1-mini-2024-09-12"
  ):  # https://platform.openai.com/docs/models/#o1
    max_tokens = 128000
  elif model_name == "o1-preview":  # https://platform.openai.com/docs/models/#o1
    max_tokens = 128000
  elif (
    model_name == "o1-preview-2024-09-12"
  ):  # https://platform.openai.com/docs/models/#o1
    max_tokens = 128000
  elif (
    model_name == "gpt-4.5-preview"
  ):  # https://platform.openai.com/docs/models/gpt-4.5-preview
    max_tokens = 128000
  elif (
    model_name == "gpt-4.1"
  ):  # https://platform.openai.com/docs/models/gpt-4.1
    max_tokens = 1048576
  elif (
    model_name == "gpt-4.1-mini"
  ):  # https://platform.openai.com/docs/models/gpt-4.1-mini
    max_tokens = 1048576
  elif (
    model_name == "gpt-4.1-nano"
  ):  # https://platform.openai.com/docs/models/gpt-4.1-nano
    max_tokens = 1048576
  elif (
    model_name == "gpt-4o-mini"
  ):  # https://platform.openai.com/docs/models/gpt-4o-mini
    max_tokens = 128000
  elif (
    model_name == "gpt-4o-mini-2024-07-18"
  ):  # https://platform.openai.com/docs/models/gpt-4o-mini
    max_tokens = 128000
  elif model_name == "gpt-4o":  # https://platform.openai.com/docs/models/gpt-4o
    max_tokens = 128000
  elif (
    model_name == "gpt-4o-2024-05-13"
  ):  # https://platform.openai.com/docs/models/gpt-4o
    max_tokens = 128000
  elif (
    model_name == "gpt-4o-2024-08-06"
  ):  # https://platform.openai.com/docs/models/gpt-4o
    max_tokens = 128000
  elif (
    model_name == "gpt-4o-2024-11-20"
  ):  # https://platform.openai.com/docs/models/gpt-4o
    max_tokens = 128000
  elif (
    model_name == "chatgpt-4o-latest"
  ):  # https://platform.openai.com/docs/models/gpt-4o
    max_tokens = 128000
  elif model_name == "gpt-4-turbo":  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 128000
  elif (
    model_name == "gpt-4-turbo-2024-04-09"
  ):  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 128000
  elif (
    model_name == "gpt-4-turbo-preview"
  ):  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 128000
  elif (
    model_name == "gpt-4-0125-preview"
  ):  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 128000
  elif (
    model_name == "gpt-4-1106-preview"
  ):  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 128000
  elif model_name == "gpt-4-32k":  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 32768
  elif (
    model_name == "gpt-3.5-turbo-16k"
  ):  # https://platform.openai.com/docs/models/gpt-3-5
    max_tokens = 16384
  elif model_name == "gpt-4":  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 8192
  elif model_name == "gpt-4-0314":  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 8192
  elif model_name == "gpt-4-0613":  # https://platform.openai.com/docs/models/gpt-4
    max_tokens = 8192
  elif (
    model_name == "gpt-3.5-turbo-0125"
  ):  # https://platform.openai.com/docs/models/gpt-3-5-turbo
    max_tokens = 16385
  elif (
    model_name == "gpt-3.5-turbo"
  ):  # https://platform.openai.com/docs/models/gpt-3-5-turbo
    max_tokens = 16385
  elif (
    model_name == "gpt-3.5-turbo-1106"
  ):  # https://platform.openai.com/docs/models/gpt-3-5-turbo
    max_tokens = 16385
  elif (
    model_name == "gpt-3.5-turbo-instruct"
  ):  # https://platform.openai.com/docs/models/gpt-3-5-turbo
    max_tokens = 4096
  else:
    max_tokens = 128000

  return max_tokens

# / def get_max_tokens_for_model(model_name):


# TODO: caching support
def run_llm_completion(
  model_name, gpt_timeout, messages, temperature=0, max_output_tokens=100
):
  is_claude = model_name.startswith('claude-')

  if is_claude:
    num_input_tokens = 0 # TODO
  else:
    num_input_tokens = num_tokens_from_messages(
      messages, model_name
    )  # TODO: a more precise token count is already provided by OpenAI, no need to recalculate it here

  max_tokens = get_max_tokens_for_model(model_name)

  print(f"num_input_tokens: {num_input_tokens} max_tokens: {max_tokens}")

  time_start = time.time()

  (response_content, finish_reason, input_tokens, output_tokens) = completion_with_backoff(
    gpt_timeout,
    model=model_name,
    messages=messages,
    n=1,
    stream=False,
    temperature=temperature,  # 1,   0 means deterministic output  # TODO: increase in case of sampling the GPT multiple times per same text
    top_p=1,
    max_tokens=max_output_tokens,
    presence_penalty=0,
    frequency_penalty=0,
    # logit_bias = None,
  )

  time_elapsed = time.time() - time_start

  too_long = finish_reason == "length" if not is_claude else finish_reason == "max_tokens"
  assert not too_long

  output_message = {"role": "assistant", "content": response_content}
  if is_claude:
    num_output_tokens = output_tokens
    num_total_tokens = input_tokens + output_tokens
  else:
    # TODO: use input_tokens, output_tokens variables
    num_output_tokens = num_tokens_from_messages(
      [output_message], model_name
    )  # TODO: a more precise token count is already provided by OpenAI, no need to recalculate it here
    num_total_tokens = num_input_tokens + num_output_tokens

  print(
    f"num_total_tokens: {num_total_tokens} num_output_tokens: {num_output_tokens} max_tokens: {max_tokens} performance: {(num_output_tokens / time_elapsed)} output_tokens/sec"
  )

  return response_content, output_message

# / def run_llm_completion(model_name, gpt_timeout, messages, temperature = 0, sample_index = 0):


def extract_int_from_text(text):    # TODO: implement extract_float_from_text as well

  result = int(''.join(c for c in text if c.isdigit() or c == "-"))
  return result

def format_float(value):
  if abs(value) < 1e-3:  # TODO: tune/config
    value = 0
  # format to have three numbers in total, regardless whether they are before or after comma
  text = "{0:.3f}".format(float(value))  # TODO: tune/config
  if text == "0.000" or text == "-0.000":
    text = "0.0"
  return text
