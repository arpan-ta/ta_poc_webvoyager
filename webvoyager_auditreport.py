import os
from getpass import getpass
import asyncio
import platform
import base64
from datetime import datetime
import argparse
from langchain import hub
import random
import time
import json
import re
from io import BytesIO
from langchain_openai import AzureChatOpenAI
from PIL import Image
from playwright.async_api import async_playwright, Page
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import chain as chain_decorator
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
import pandas as pd
import openpyxl

import warnings
warnings.filterwarnings('ignore')

base_path = "/home/anisha/Langraph_web_voyager/ta_poc_webvoyager/"
config_path = "/home/anisha/Langraph_web_voyager/ta_poc_webvoyager/config.json"
with open(config_path, 'r') as config_file:
    config_details = json.load(config_file)
# Load config values
openai_api_endpoint = config_details['config_list'][0]['azure_endpoint']

# API version e.g. "2023-07-01-preview"
openai_api_version = config_details['config_list'][0]['api_version']

model_name = config_details['config_list'][0]['model']
# The API key for your Azure OpenAI resource.
openai_api_key = config_details['config_list'][0]['api_key']
# The APItype for your Azure OpenAI resource.
openai_api_type = config_details['config_list'][0]['api_type']

llm = AzureChatOpenAI(
    openai_api_version=openai_api_version,
    azure_endpoint=openai_api_endpoint,
    openai_api_key=openai_api_key,
    openai_api_type=openai_api_type,
    temperature=0.2,
    max_tokens=4096,
    timeout=120,
    max_retries=10,
    model=model_name,
    # other params...
)

def simulate_human_typing(text):
    start = 0
    word_len = len(text)+1
    rand_int = random.randint(1,3)
    end = start + min(rand_int, word_len)

    while True:
        output =  text[start:end]

        start = end
        end = min(start+rand_int, word_len+1)

        yield output

        if end>=word_len+1:
            break


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool


async def click(state: AgentState):
    # - Click [Numerical_Label]
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    # TODO: In the paper, they automatically parse any downloaded PDFs
    # We could add something similar here as well and generally
    # improve response format.
    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    for key_strokes in simulate_human_typing(text_content):
        dynamic_delay = random.random()*100
        await page.keyboard.type(key_strokes, delay=dynamic_delay)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def go_to_url(state: AgentState):
    page = state["page"]
    url_args = state["prediction"]["args"]
    if url_args is None or len(url_args) != 1:
        return "Failed to go to URL due to incorrect arguments."
    
    url = url_args[0]
    print("Trying to print",url)
    if not url.startswith("http"):
        url = "https://" + url + "/" # Make sure the URL starts with http or https
    
    await page.goto(url)
    return f"Navigated to {url}"


async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


async def to_homepage(state: AgentState):
    page = state["page"]
    await page.goto("https://www.chromium.org")
    return "Navigated to the default home page."

async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

async def summarize_image(state: AgentState):
    """Summarizes an image with a short context of what is displayed."""
    page = state["page"]
    screenshot = await page.screenshot()
    image_url = base64.b64encode(screenshot)
    if not image_url:
        return {**state, "observation": "No image provided"}
    print('Screenshot & Summarize image')
    # Load Image
    image_data = base64.b64decode(image_url)#.decode("utf-8")
    # img = Image.open(BytesIO(image_data))
    img = Image.open(BytesIO(image_data))


    random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    img_file = os.path.expanduser(f'{base_path}image_{random_id}.jpg')
    img.save(img_file)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(img_file)

    # Send image to GPT-4o for summarization
    messages = [
        SystemMessage(content="You are an AI that summarizes images concisely."),
        HumanMessage(content="""Describe the employee's status. Assess if the employee is active and has access flag"""),
        HumanMessage(content= [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ])]
    # https: // platform.openai.com / docs / guides / vision
    #https://github.com/langchain-ai/langchain/discussions/23374
    # response = llm(messages)
    # print(response.content)
    # Store the summary in the agent state
    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=messages)

    output_parser = StrOutputParser()
    chain = chat_prompt_template | llm | output_parser
    result = chain.invoke({})

    txt_file = os.path.expanduser(f"{base_path}result_{random_id}.txt")
    with open(txt_file, "w") as file:
        file.write(f"Image saved at: {img_file}\n")
        file.write(result)

    print(result)


    # Extract details using regular expressions
    name_match = re.search(r'The employee, (.*?), has a status', result)
    status_match = re.search(r'status of "(.*?)"', result)
    access_flag_match = re.search(r'Access Flag" set to "(.*?)"', result)
    
    
    employee_name = name_match.group(1) if name_match else "N/A"
    status = status_match.group(1) if status_match else "N/A"
    access_flag = access_flag_match.group(1) if access_flag_match else "N/A"
    

    # Prepare data for the Excel report
    report_data = {
        "Employee Name": employee_name,
        "Status": status,
        "Access Flag": access_flag,
        "User Comments": result # The second line of the result
    }
    
    # Define the Excel file path
    excel_file = os.path.join(base_path, "audit_report.xlsx")
    
    # Check if the Excel file exists
    if os.path.exists(excel_file):
        # Load existing data
        df_existing = pd.read_excel(excel_file)
        # Append new data
        df = pd.concat([df_existing, pd.DataFrame([report_data])], ignore_index=True)
    else:
        # Create a new DataFrame
        df = pd.DataFrame([report_data])
    
    # Save the DataFrame to Excel
    df.to_excel(excel_file, index=False)
    print('Audit report updated')
    
    return f"{result}"#, "observation": response.content}



# Some javascript we will run on each step
# to take a screenshot of the page, select the
# elements to annotate, and add bounding boxes
with open("mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            asyncio.sleep(3)
    screenshot = await page.screenshot()
    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }


async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    answer_prefix = "Answer"
    if text.strip().split("\n")[-1].startswith(answer_prefix):
        action_block = text.strip().split("\n")[-1]
        action_str = action_block[len(answer_prefix) :]
        split_output = action_str.split(" ", 1)
        if len(split_output) == 1:
            action, action_input = split_output[0], None
        else:
            action, action_input = split_output
        return {"action": action, "args": action_input}
    
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    
    action_block = text.strip().split("\n")[-1]
    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


# Will need a later version of langchain to pull
# this image prompt template
#prompt = hub.pull("avg/login-agent")
prompt = hub.pull("avg/web-voyager_adapted_signin")

agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)


def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}




graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "Navigate": go_to_url,
    "GoBack": go_back,
    "HomePage": to_homepage,
    "summarize_image": summarize_image,
}


for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        # The lambda ensures the function's string output is mapped to the "observation"
        # key in the AgentState
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    # Always return to the agent (by means of the update-scratchpad node)
    graph_builder.add_edge(node_name, "update_scratchpad")


def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"]["action"]
    # if action.lower().startswith("answer"):
    #     return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_node(
        'Answer',
        RunnableLambda(summarize_image) | (lambda observation: {"observation": observation}),
    )
graph_builder.add_conditional_edges("agent", select_tool)
graph_builder.add_edge("Answer", END)

graph = graph_builder.compile()


async def main(keyword="github"):
    async with async_playwright() as p:
        if keyword == "github":
            prompt = "Login to github.com with spartan07 as username and  xxxxx as password"
        elif keyword == "docker":
            prompt = "Login to docker.com with arpan92 as username and xxxx as password"
        elif keyword == "local":
            prompt = """1.Login to http://localhost:8000/employee_portal.html and login with admin as username and password as password.
            2.After logging in click on employee lookup tab and enter employee id 12345 and click on Search.
            3.Comment on the employee status and the active flag."""
        elif keyword == "github-ta":
            prompt = """.Login to https://anishadesai-ta.github.io/companydemo/ and login with admin as username and password as password123.
            2.After logging in click on employee lookup tab and search for employee David Miller and generate search results
            3.Answer with David Miller's employee status and access flag"""
        print(f"Given Prompt: {prompt}")
        print("Initiating agent shortly....")
        time.sleep(8)
        # We will set headless=False so we can watch the agent navigate the web.
        browser = await p.chromium.launch(headless=False, args=None)
        page = await browser.new_page()
        await page.goto("https://www.chromium.org")

        res = await call_agent(prompt,
            page,
        )

async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    async for event in event_stream:
        # We'll display an event stream here
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        #display.clear_output(wait=False)
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        print("\n".join(steps))
        #display.display(display.Image(base64.b64decode(event["agent"]["img"])))
        # if "answer" in action.lower():
        #     final_answer = action_input[0]
        #     break
    return final_answer


# Run the main function

if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="Run web automation task with a specified prompt.")
    parser.add_argument("--keyword", type=str, default = "local", help="The prompt to run.")

    args= parser.parse_args()
    print(f"Received argument: {args.keyword}")
    asyncio.run(main(args.keyword))