from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI
from anthropic import AsyncAnthropic, Anthropic
from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI
import asyncio
from asyncio import Semaphore
from datetime import datetime, timezone
import time

load_dotenv()

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the most fun thing to do in San Francisco?",
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 200000,
                 context_lengths_num_intervals = 35,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 35,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='gpt-4-1106-preview',
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        if model_provider not in ["OpenAI", "Anthropic"]:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
        if model_provider == "Anthropic" and "claude" not in model_name:
            raise ValueError("If the model provider is 'Anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

        if not self.openai_api_key and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")
        else:
            self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')

        if self.model_provider == "Anthropic":
            if not self.anthropic_api_key and not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("Either anthropic_api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env.")
            else:
                self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            
        if not self.model_name:
            raise ValueError("model_name must be provided.")
        
        if model_provider == "OpenAI":
            self.model_to_test = AsyncOpenAI(api_key=self.openai_api_key)
            self.enc = tiktoken.encoding_for_model(self.model_name)
        elif model_provider == "Anthropic":
            self.model_to_test = AsyncAnthropic(api_key=self.anthropic_api_key)
            self.enc = Anthropic().get_tokenizer()
        
        self.model_to_test_description = model_name
        self.evaluation_model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = self.openai_api_key)

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def generate_prompt(self, context):
        if self.model_provider == "Anthropic":
            return f"""Human: You are a close-reading bot with a great memory who answers questions for users. I'm going to give you the text of some essays. Amidst these essays ("the haystack") I've inserted a sentence ("the needle") that contains an answer to the user's question. Here's the question:
                <question>{self.retrieval_question}</question>
                Here's the text of the essays. The answer appears in it somewhere.
                <haystack>
                {context}
                </haystack>
                Now that you've read the context, please answer the user's question, repeated one more time for ease of reference:
                <question>{self.retrieval_question}</question>

                To do so, first find the sentence from the haystack that contains the answer (there is such a sentence, I promise!) and put it inside <most_relevant_sentence> XML tags. Then, put your answer in <answer> tags. Base your answer strictly on the context, without reference to outside information. Thank you.

                Assistant:"""
        elif self.model_provider == "OpenAI":
            # Generate the prompt for the Anthropic model
            # Replace the following line with the appropriate prompt structure
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings"
                }
            ]

    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)

        test_start_time = time.time()

        # Go see if the model can answer the question to pull out your random fact
        if self.model_provider == "OpenAI":
            response = await self.model_to_test.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=300,
                temperature=0
            )
            response = response.choices[0].message.content
        elif self.model_provider == "Anthropic":
            response = await self.model_to_test.completions.create(
                model=self.model_name,
                max_tokens_to_sample=300,
                prompt=prompt,
                temperature=0
            )
            response = response.completion

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        score = self.evaluate_response(response)


        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results'):
                os.makedirs('results')

            # Save the result to file for retesting
            with open(f'results/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider == "OpenAI":
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def evaluate_response(self, response):
        accuracy_criteria = {
            "accuracy": """
            Score 1: The answer is completely unrelated to the reference.
            Score 3: The answer has minor relevance but does not align with the reference.
            Score 5: The answer has moderate relevance but contains inaccuracies.
            Score 7: The answer aligns with the reference but has minor omissions.
            Score 10: The answer is completely accurate and aligns perfectly with the reference.
            Only respond with a numberical score
            """
        }

        # Using GPT-4 to evaluate
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=accuracy_criteria,
            llm=self.evaluation_model,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.needle,

            # The question asked
            input=self.retrieval_question,
        )

        return int(eval_result['score'])

    def get_context_length_in_tokens(self, context):
        if self.model_provider == "OpenAI":
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return len(self.enc.encode(context).ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider == "OpenAI":
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider == "OpenAI":
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())

if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = LLMNeedleHaystackTester()

    ht.start_test()