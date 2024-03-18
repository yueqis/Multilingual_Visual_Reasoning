import numpy as np
import asyncio
import nest_asyncio

nest_asyncio.apply()

from .step_interpreters import register_step_interpreters, parse_step

# get all statements
def get_statement_dict(marvl_rows, length):
    statement_dict = {}
    for idx in range(length):
        statement = marvl_rows[idx]['caption']
        unique_id = marvl_rows[idx]['unique_id']
        if (unique_id not in statement_dict): statement_dict[unique_id] = statement
    return statement_dict

class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')

class ProgramInterpreter:
    def __init__(self,dataset='nlvr'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self,prog_step,inspect):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        #print(step_name)
        return self.step_interpreters[step_name].execute(prog_step,inspect)

    def execute(self,prog,init_state,inspect=False):
        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        html_str = '<hr>'
        for prog_step in prog_steps:
            if inspect:
                step_output, step_html = self.execute_step(prog_step,inspect)
                html_str += step_html + '<hr>'
            else:
                step_output = self.execute_step(prog_step,inspect)

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state

class ProgramGenerator():
    def __init__(self,prompter,client,temperature=0.7,top_p=0.5,prob_agg='mean'):
        self.prompter = prompter
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg
        self.client = client

    def compute_prob(self,response):
        eos = '\n'
        for i,token in enumerate(response.details.tokens):
            if i!=0 and token.text==eos and response.details.tokens[i-1].text == eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError
        
        texts = ''.join([token.text for token in response.details.tokens[:i]])
        logprobs = [token.logprob for token in response.details.tokens[:i]]        
        return (texts, np.exp(agg_fn(logprobs)))

    def generate(self,inputs):
        response = self.client.generate(
            prompt=self.prompter(inputs),
            max_new_tokens=512,
            temperature=self.temperature,
            top_p=self.top_p,
            decoder_input_details=True
        )

        (texts,_) = self.compute_prob(response)
        prog = texts.lstrip('\n').rstrip('\n')
        return prog
    
    def get_prog(self, response):
        eos = '\n'
        for i,token in enumerate(response.details.tokens):
            if i!=0 and token.text==eos and response.details.tokens[i-1].text == eos:
                break
        
        texts = ''.join([token.text for token in response.details.tokens[:i]])
        return texts
    
    def generate_async(self, inputs_list):
        async def batch_generate(SAMPLES):
            return await asyncio.gather(*[self.client.generate(prompt=sample, max_new_tokens=512, temperature=self.temperature, top_p=self.top_p, decoder_input_details=True) for sample in SAMPLES])
        results = asyncio.run(batch_generate([self.prompter(inputs) for inputs in inputs_list]))
        return [self.get_prog(result).lstrip('\n').rstrip('\n') for result in results]