from ragas.testset.persona import Persona

from dataclasses import dataclass
import typing as t
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

import synthetic_data_generation_setup_multihop

import asyncio

from ragas.testset.graph import KnowledgeGraph
from langchain_community.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from dotenv import load_dotenv
import os

project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

persona_list = [
    Persona(
        name="Literature Archivist",
        role_description="A researcher cataloguing the structural layout of the Sherlock Holmes novels including parts, chapters, and publication details.",
    ),
    Persona(
        name="Story Analyst",
        role_description="A literary analyst examining events, clues, timelines, and plot progressions throughout the Sherlock Holmes novels.",
    ),
    Persona(
        name="Consulting Detective Psychologist",
        role_description="A specialist analyzing character traits, motivations, and behaviors of Holmes, Watson, and other recurring figures.",
    ),
    Persona(
        name="Crime Pattern Statistician",
        role_description="A researcher identifying recurring crime types, investigative patterns, and clue structures across the novels.",
    ),
    Persona(
        name="Prose Critic",
        role_description="A critic studying Arthur Conan Doyle's writing style, tone, and narrative decisions across the novels.",
    )
]

@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):

    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph,
        persona_list,
        callbacks,
    ) -> t.List[MultiHopScenario]:

        # query and get (node_a, rel, node_b) to create multi-hop queries
        results = knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: (
                True if rel.type == "keyphrases_overlap" else False
            )
        )

        num_sample_per_triplet = max(1, n // len(results))

        scenarios = []
        for triplet in results:
            if len(scenarios) < n:
                node_a, node_b = triplet[0], triplet[-1]
                overlapped_keywords = triplet[1].properties["overlapped_items"]
                if overlapped_keywords:

                    # match the keyword with a persona for query creation
                    themes = list(dict(overlapped_keywords).keys())
                    prompt_input = ThemesPersonasInput(
                        themes=themes, personas=persona_list
                    )
                    persona_concepts = (
                        await self.theme_persona_matching_prompt.generate(
                            data=prompt_input, llm=self.llm, callbacks=callbacks
                        )
                    )

                    overlapped_keywords = [list(item) for item in overlapped_keywords]

                    # prepare and sample possible combinations
                    base_scenarios = self.prepare_combinations(
                        [node_a, node_b],
                        overlapped_keywords,
                        personas=persona_list,
                        persona_item_mapping=persona_concepts.mapping,
                        property_name="keyphrases",
                    )

                    # get number of required samples from this triplet
                    base_scenarios = self.sample_diverse_combinations(
                        base_scenarios, num_sample_per_triplet
                    )

                    scenarios.extend(base_scenarios)

        return scenarios

    
# rebuild kg
the_kg, the_llm = synthetic_data_generation_setup_multihop.process_before_test()
the_kg.save(path="/Users/mertozdemir/Documents/Tech Narts/Orientation Training/Sherlock-Project/Advanced-RAG/data/whole_corpus_kg_multihop.json")

# use pre-saved kg
"""the_kg = KnowledgeGraph.load("/Users/mertozdemir/Documents/Tech Narts/Orientation Training/Sherlock-Project/Advanced-RAG/data/whole_corpus_kg_multihop.json")
openai_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)
the_llm = LangchainLLMWrapper(openai_llm)"""


query = MyMultiHopQuery(llm=the_llm)

async def main():
    scenarios = await query.generate_scenarios(
        n=100, knowledge_graph=the_kg, persona_list=persona_list
    )
    test_items = []

    for sc in scenarios:
        sample = await query.generate_sample(scenario=sc)
        
        test_items.append({
            "query": sample.user_input,
            "reference": sample.reference,
            "scenario": sc
        })

    return test_items


results = asyncio.run(main())

output_path = "ragas_test_items_multihop.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for i, r in enumerate(results, start=1):
        f.write(f"=== TEST ITEM {i} ===\n")
        f.write(f"QUERY:\n{r['query']}\n\n")
        f.write(f"REFERENCE:\n{r['reference']}\n\n")
        f.write("SCENARIO:\n")
        f.write(str(r["scenario"]))  # serialize scenario object
        f.write("\n\n------------------------------\n\n")

print(f"Saved results to {output_path}")
