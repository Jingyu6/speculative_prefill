import atexit

import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM

from rag_baseline.rag_utils import (retrieve_query_fn,
                                    retrieve_relevant_sentences)


class RagLlama:
    def __init__(
        self, 
        llama_model_name: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct', 
        percentage: float = 0.5
    ) -> None:
        self.llama = LlamaForCausalLM.from_pretrained(
            llama_model_name, 
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.percentage = percentage

        # stats related stuffs
        self.reset_stats()
        atexit.register(self.print_stats)

    def reset_stats(self):
        self.num_queries = 0
        self.ratio = 0

    def print_stats(self):
        if self.num_queries == 0:
            print("Currently no processed queries. ")
            avg_ratio = None
        else:
            avg_ratio = self.ratio / self.num_queries
            print(f"Processed {self.num_queries} queries with avg {avg_ratio * 100:.2f}% keep ratio.")
        return self.num_queries, avg_ratio

    def update_stats(self, ratio):
        self.num_queries += 1
        self.ratio += ratio

    @torch.inference_mode
    def generate(
        self, 
        context: str, 
        input: str, 
        prompt_format: str, 
        dataset_name: str, 
        max_gen: int, 
        apply_chat_template: bool
    ): 
        # query used for retrieval
        query = retrieve_query_fn(dataset_name=dataset_name)(input)

        # calculate original length of the prompt
        ori_prompt = prompt_format.format(input=input, context=context)
        if apply_chat_template:
            messages = [{'role': 'user', 'content': ori_prompt}]
            ori_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        inputs = self.tokenizer([ori_prompt], return_tensors='pt')
        ori_len = inputs['input_ids'].shape[1]

        # calculate retrieval length
        ret_len = int(ori_len * self.percentage)

        # rag part
        ret_context = retrieve_relevant_sentences([context], [query], token_budgets=[ret_len])[0]

        # assemble new prompt
        rag_prompt = prompt_format.format(input=input, context=ret_context)
        if apply_chat_template:
            messages = [{'role': 'user', 'content': rag_prompt}]
            rag_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        inputs = self.tokenizer([rag_prompt], return_tensors='pt')
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        final_len = input_ids.shape[1]
        
        gen_config = GenerationConfig(
            do_sample=False, 
            eos_token_id=128009, 
            pad_token_id=128009
        )

        # update stats
        self.update_stats(final_len / ori_len)

        outputs = self.llama.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  
            max_new_tokens=max_gen, 
            return_dict_in_generate=True, 
            generation_config=gen_config
        )

        generated_tokens = outputs.sequences[:, final_len:]

        return self.tokenizer.decode(generated_tokens.tolist()[0], skip_special_tokens=True)

if __name__ == "__main__":
    model = RagLlama()
    input_text = """
During World War I and the 1920s there was a major expansion in industry. The availability of jobs attracted African Americans from the Southern United States. Between 1910 and 1930, the African American population of Chicago increased dramatically, from 44,103 to 233,903.[68] This Great Migration had an immense cultural impact, called the Chicago Black Renaissance, part of the New Negro Movement, in art, literature, and music.[69] Continuing racial tensions and violence, such as the Chicago race riot of 1919, also occurred.[70]
The ratification of the 18th amendment to the Constitution in 1919 made the production and sale (including exportation) of alcoholic beverages illegal in the United States. This ushered in the beginning of what is known as the gangster era, a time that roughly spans from 1919 until 1933 when Prohibition was repealed. The 1920s saw gangsters, including Al Capone, Dion O'Banion, Bugs Moran and Tony Accardo battle law enforcement and each other on the streets of Chicago during the Prohibition era.[71] Chicago was the location of the infamous St. Valentine's Day Massacre in 1929, when Al Capone sent men to gun down members of a rival gang, North Side, led by Bugs Moran.[72]
Chicago tenants picket against rent increases (March 1920)
From 1920 to 1921, the city was affected by a series of tenant rent strikes, which lead to the formation of the Chicago Tenants Protective association, passage of the Kessenger tenant laws, and of a heat ordinance that legally required flats to be kept above 68 °F during winter months by landlords.[73][74][75][76][77][78]
Chicago was the first American city to have a homosexual-rights organization. The organization, formed in 1924, was called the Society for Human Rights. It produced the first American publication for homosexuals, Friendship and Freedom. Police and political pressure caused the organization to disband.[79]
Men outside a soup kitchen during the Great Depression (1931)
The Great Depression brought unprecedented suffering to Chicago, in no small part due to the city's heavy reliance on heavy industry. Notably, industrial areas on the south side and neighborhoods lining both branches of the Chicago River were devastated; by 1933 over 50% of industrial jobs in the city had been lost, and unemployment rates amongst blacks and Mexicans in the city were over 40%. The Republican political machine in Chicago was utterly destroyed by the economic crisis, and every mayor since 1931 has been a Democrat.[80]
From 1928 to 1933, the city witnessed a tax revolt, and the city was unable to meet payroll or provide relief efforts. The fiscal crisis was resolved by 1933, and at the same time, federal relief funding began to flow into Chicago.[80] Chicago was also a hotbed of labor activism, with Unemployed Councils contributing heavily in the early depression to create solidarity for the poor and demand relief; these organizations were created by socialist and communist groups. By 1935 the Workers Alliance of America begun organizing the poor, workers, the unemployed. In the spring of 1937 Republic Steel Works witnessed the Memorial Day massacre of 1937 in the neighborhood of East Side.
In 1933, Chicago Mayor Anton Cermak was fatally wounded in Miami, Florida, during a failed assassination attempt on President-elect Franklin D. Roosevelt. In 1933 and 1934, the city celebrated its centennial by hosting the Century of Progress International Exposition World's Fair.[81] The theme of the fair was technological innovation over the century since Chicago's founding.[82]
1940 to 1979
The Chicago Picasso (1967) inspired a new era in urban public art.
During World War II, the city of Chicago alone produced more steel than the United Kingdom every year from 1939 – 1945, and more than Nazi Germany from 1943 – 1945.[83]
Protesters in Grant Park outside the 1968 Democratic National Convention
The Great Migration, which had been on pause due to the Depression, resumed at an even faster pace in the second wave, as hundreds of thousands of blacks from the South arrived in the city to work in the steel mills, railroads, and shipping yards.[84]
On December 2, 1942, physicist Enrico Fermi conducted the world's first controlled nuclear reaction at the University of Chicago as part of the top-secret Manhattan Project. This led to the creation of the atomic bomb by the United States, which it used in World War II in 1945.[85]
Mayor Richard J. Daley, a Democrat, was elected in 1955, in the era of machine politics. In 1956, the city conducted its last major expansion when it annexed the land under O'Hare airport, including a small portion of DuPage County.[86]
By the 1960s, white residents in several neighborhoods left the city for the suburban areas – in many American cities, a process known as white flight – as Blacks continued to move beyond the Black Belt.[87] While home loan discriminatory redlining against blacks continued, the real estate industry practiced what became known as blockbusting, completely changing the racial composition of whole neighborhoods.[88] Structural changes in industry, such as globalization and job outsourcing, caused heavy job losses for lower-skilled workers. At its peak during the 1960s, some 250,000 workers were employed in the steel industry in Chicago, but the steel crisis of the 1970s and 1980s reduced this number to just 28,000 in 2015. In 1966, Martin Luther King Jr. and Albert Raby led the Chicago Freedom Movement, which culminated in agreements between Mayor Richard J. Daley and the movement leaders.[89]
Two years later, the city hosted the tumultuous 1968 Democratic National Convention, which featured physical confrontations both inside and outside the convention hall, with anti-war protesters, journalists and bystanders being beaten by police.[90] Major construction projects, including the Sears Tower (now known as the Willis Tower, which in 1974 became the world's tallest building), University of Illinois at Chicago, McCormick Place, and O'Hare International Airport, were undertaken during Richard J. Daley's tenure.[91] In 1979, Jane Byrne, the city's first female mayor, was elected. She was notable for temporarily moving into the crime-ridden Cabrini-Green housing project and for leading Chicago's school system out of a financial crisis.[92]
1980 to present
In 1983, Harold Washington became the first black mayor of Chicago. Washington's first term in office directed attention to poor and previously neglected minority neighborhoods. He was re‑elected in 1987 but died of a heart attack soon after.[93] Washington was succeeded by 6th ward alderperson Eugene Sawyer, who was elected by the Chicago City Council and served until a special election.
Richard M. Daley, son of Richard J. Daley, was elected in 1989. His accomplishments included improvements to parks and creating incentives for sustainable development, as well as closing Meigs Field in the middle of the night and destroying the runways. After successfully running for re-election five times, and becoming Chicago's longest-serving mayor, Richard M. Daley declined to run for a seventh term.[94][95]
In 1992, a construction accident near the Kinzie Street Bridge produced a breach connecting the Chicago River to a tunnel below, which was part of an abandoned freight tunnel system extending throughout the downtown Loop district. The tunnels filled with 250 million US gallons (1,000,000 m3) of water, affecting buildings throughout the district and forcing a shutdown of electrical power.[96] The area was shut down for three days and some buildings did not reopen for weeks; losses were estimated at $1.95 billion.[96]
On February 23, 2011, Rahm Emanuel, a former White House Chief of Staff and member of the House of Representatives, won the mayoral election.[97] Emanuel was sworn in as mayor on May 16, 2011, and won re-election in 2015.[98] Lori Lightfoot, the city's first African American woman mayor and its first openly LGBTQ mayor, was elected to succeed Emanuel as mayor in 2019.[99] All three city-wide elective offices were held by women (and women of color) for the first time in Chicago history: in addition to Lightfoot, the city clerk was Anna Valencia and the city treasurer was Melissa Conyears-Ervin.[100]
On May 15, 2023, Brandon Johnson assumed office as the 57th mayor of Chicago.
How many times has "Chicago" been mentioned in the above text?
"""
    print(model.generate(input_text, max_gen=32, apply_chat_template=True))