import json
from typing import Dict , List , Optional ,Any
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import requests
from bs4 import BeautifulSoup

@dataclass
class SearchResult:
    query: str
    content: str
    relevance_score: float
    timestamp: str
    source: Optional[str]=None



@dataclass
class AgentMemory:
    goal: str
    search_history: List[Dict[str,Any]]
    knowledge_base: List[SearchResult]
    reasoning_log: List[str]

    def add_search(self, query: str , results: List[Dict[str,Any]]):
        self.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result_count": len(results),

        })

    def add_knowledge(self, result:SearchResult):
        self.knowledge_base.append(result)

    def add_reasoning(self, reasoning:str):

        self.reasoning_log.append({
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning
        })


# noinspection PyArgumentList
class AgenticSearchAgent:

    def __init__(self,model_name: str="Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          torch_dtype=torch.float16,
                                                          device_map="auto"
                                                          )

        self.memory : Optional[AgentMemory] = None
        self.max_length = 2048

    def generate_response(self, prompt: str , max_tokens: int = 1000) -> str:

        messages =[
            {"role": "system", "content": "You are a helpful AI assistant that provides clear, concise answers."},
            {"role": "user", "content": prompt}

        ]

        text= self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids ,output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def initialize_memory(self, goal: str):
        self.memory = AgentMemory(
            goal=goal,
            search_history=[],
            knowledge_base=[],
            reasoning_log=[]
        )

    def decide_search_necessity(self, goal: str )-> Dict[str, Any]:

        search_keywords = [
            'current', 'latest', 'recent', 'today', 'now', 'price', 'cost',
            'weather', 'news', 'who is', 'what is the', '2024', '2025',
            'happening', 'update', 'this year', 'this month', 'this week'
        ]

        goal_lower = goal.lower()
        should_search_heuristic = any(keyword in goal_lower for keyword in search_keywords)

        if should_search_heuristic:
            decision = {
                "search_needed": True,
                "reasoning": "Query contains time-sensitive or current information keywords",
                "information_needed": ["current data", "latest information"],
                "confidence": 0.9
            }
        else:

            prompt = f"""Decide if this question needs web search. 

                        Question: {goal}

                        Does this need current/live data from the web? Answer only YES or NO, then give a brief reason.

                        Format:
                        DECISION: YES/NO
                        REASON: brief explanation"""

            response_text = self.generate_response(prompt, max_tokens=200)

            # Simple parsing
            if "YES" in response_text.upper() or "DECISION: YES" in response_text.upper():
                search_needed = True
            elif "NO" in response_text.upper() or "DECISION: NO" in response_text.upper():
                search_needed = False
            else:

                search_needed = True

            decision = {
                "search_needed": search_needed,
                "reasoning": response_text.strip()[:200],
                "information_needed": ["relevant information"],
                "confidence": 0.7
            }

        if self.memory:
            self.memory.add_reasoning(
                f"Search Decision: {decision['search_needed']} - {decision['reasoning']}"
            )

        return decision

    def generate_search_queries(self, goal: str, information_needed: List[str]) -> List[str]:

        queries= []

        main_query = goal.strip().replace("?", "").replace("What is", "").replace("Tell me about", "").strip()
        if main_query:
            queries.append(main_query)

        if "latest" not in goal.lower() and "current" not in goal.lower():
            queries.append(f"{main_query} latest")
            queries.append(f"{main_query} 2025")

        queries = queries[:3]

        try:
            prompt = f"""Generate 2 good search queries for: {goal}

                        Just list the queries, one per line, no numbering or extra text."""

            response_text = self.generate_response(prompt, max_tokens=200)


            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            lines = [line for line in lines if len(line) > 5 and len(line) < 100]


            for line in lines[:2]:

                line = line.replace("1.", "").replace("2.", "").replace("-", "").strip()
                if line and line not in queries:
                    queries.append(line)

        except:
            pass

        if not queries:
            queries = [goal]

        result = {"queries": queries[:3], "reasoning": "Direct query generation"}

        return result["queries"]

    def perform_web_search(self, query: str)-> List[Dict[str, Any]]:
        try:
            print(f"   Searching DuckDuckGo for: {query}")

            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            print(f"   Response status: {response.status_code}")

            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            search_results = soup.find_all('div', class_='result')[:5]  
            print(f"   Found {len(search_results)} search result elements")

            for result in search_results:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')

                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append({
                        "content": f"Title: {title}\nSnippet: {snippet}\nURL: {link}",
                        "type": "search_result",
                        "query": query,
                        "url": link
                    })
                    print(f"   Added result: {title[:50]}...")

            if not results:
                print(f"   No results found, creating fallback")
                results = [{
                    "content": f"No specific results found for query: {query}. Search was attempted but returned no results.",
                    "type": "text",
                    "query": query
                }]

            if self.memory:
                self.memory.add_search(query, results)

            print(f"   Returning {len(results)} results")
            return results

        except Exception as e:
            print(f"   Search error: {e}")
            return [{"content": f"Search failed: {str(e)}", "type": "error", "query": query}]

            return results

        except Exception as e:
            print(f"Search error: {e}")
            return [{"content": f"Search failed: {str(e)}", "type": "error", "query": query}]


    def evaluate_search_results(self,goal: str, query, results: List[Dict[str, Any]]) -> Dict[str, Any]:

        if not results or len(results) == 0:
            evaluation = {
                "relevance_score": 0.0,
                "useful_information": [],
                "need_more_search": True,
                "key_takeaways": "No results found",
                "gaps": ["Need search results"]
            }

        else:
            useful_info = []
            for result in results[:3]:
                content = result.get('content', '')
                if content and len(content) > 20:
                    useful_info.append(content[:200])

            evaluation = {
                "relevance_score": 0.8,
                "useful_information": useful_info,
                "need_more_search": False,
                "key_takeaways": f"Found {len(results)} results for: {query}",
                "gaps": []
            }

        if self.memory:
            self.memory.add_reasoning(
                f"Search Evaluation - Score: {evaluation['relevance_score']} - {evaluation['key_takeaways']}"
            )

            for info in evaluation.get("useful_information", []):
                self.memory.add_knowledge(SearchResult(
                    query=query,
                    content=info,
                    relevance_score=evaluation["relevance_score"],
                    timestamp=datetime.now().isoformat()
                ))

        return evaluation

    def synthesize_answer(self, goal: str)-> str:

        if not self.memory:
            return "No memory initialized"

        knowledge_base = self.memory.knowledge_base
        search_history = self.memory.search_history

        if not knowledge_base and not search_history:
            prompt = f"""Answer this question concisely: {goal}"""
            return self.generate_response(prompt, max_tokens=600)

        knowledge_summary = "\n".join([
            f"- {kb.content[:300]}"
            for kb in knowledge_base[:5]
        ])

        prompt = f"""Based on this web search information, answer the question.

                Question: {goal}

                Information from web searches:
                {knowledge_summary}

                Provide a clear, accurate answer based on the search results above."""

        return self.generate_response(prompt, max_tokens=600)

    def process_goal(self, goal: str, max_searches: int = 3) -> Dict[str, Any]:
        """
        Main agentic workflow: process a goal from start to finish.
        """
        print(f"\n Processing goal: {goal}\n")

        self.initialize_memory(goal)

        print(" Deciding if web search is needed...")
        decision = self.decide_search_necessity(goal)
        print(f"   Decision: {'Search needed' if decision['search_needed'] else 'No search needed'}")
        print(f"   Reasoning: {decision['reasoning']}\n")

        if not decision["search_needed"]:
            print(" Generating answer from existing knowledge...\n")
            return {
                "goal": goal,
                "search_performed": False,
                "answer": self.synthesize_answer(goal),
                "memory": asdict(self.memory) if self.memory else {}
            }

        print(" Generating search queries...")
        queries = self.generate_search_queries(goal, decision.get("information_needed", []))
        print(f"   Generated {len(queries)} queries: {queries}\n")

        search_count = 0
        for query in queries:
            if search_count >= max_searches:
                break

            print(f"🌐 Searching: {query}")
            results = self.perform_web_search(query)
            print(f"   Retrieved {len(results)} results")

            print(" Evaluating results...")
            evaluation = self.evaluate_search_results(goal, query, results)
            print(f"   Relevance: {evaluation['relevance_score']:.2f}")
            print(f"   Key takeaways: {evaluation['key_takeaways'][:100]}...")

            search_count += 1

            if not evaluation.get("need_more_search", False):
                print("    Sufficient information gathered\n")
                break
            else:
                print(f"    Need more information: {evaluation.get('gaps', [])}\n")

        print(" Synthesizing final answer...\n")
        final_answer = self.synthesize_answer(goal)

        return {
            "goal": goal,
            "search_performed": True,
            "searches_count": search_count,
            "answer": final_answer,
            "memory": asdict(self.memory) if self.memory else {}
        }


def main():

    agent = AgenticSearchAgent()


if __name__ == "__main__":

    main()

