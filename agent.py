from __future__ import annotations as _annotations

import asyncio
import logging
import os
from asyncio import Lock
from dataclasses import dataclass

import httpx
from httpx import AsyncClient, Timeout
from pydantic_ai import Agent, ModelRetry, RunContext

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


@dataclass
class Deps:
    client: AsyncClient
    tfl_api_key: str | None


API_CALL_LOCK = Lock()  # We only execute 1 api call at a time to avoid TFL API errors
STOP_TYPES_CACHE = {}

tfl_search = Agent(
    'openai:gpt-4o',
    system_prompt=(
        'You are an agent that can help plan journeys on Transport for London (TFL). '
        'Use the stop_point apis to get information about TFL stop points, '
        'then use the `journey_planner` tool to return the potential journeys between the relevant two stop points. '
        'Be concise, reply with one sentence.'
    ),
    deps_type=Deps,
    retries=2,
)


async def get_tfl_api(ctx: RunContext[Deps], *args, **kwargs) -> httpx.Response:
    app_key = ctx.deps.tfl_api_key
    return await ctx.deps.client.get(
        *args,
        **kwargs,
        headers=dict(**kwargs.get("headers", {}), app_key=app_key),
    )


@tfl_search.tool
async def stop_point_types(ctx: RunContext[Deps]) -> dict[str, dict]:
    """ Returns a list of all stop point types """
    logger.info("Retrieving stop types")
    r = await get_tfl_api(ctx, 'https://api.tfl.gov.uk/StopPoint/meta/stoptypes')
    r.raise_for_status()
    return {"stop_point_types": r.json()}


@tfl_search.tool
async def stop_point_list(ctx: RunContext[Deps], stop_point_type: str) -> list[dict[str, str]]:
    """ Returns a list of stop points for a given stop point type """
    async with API_CALL_LOCK:
        logger.info(f"Retrieving stop points for type {stop_point_type}")
        r = await get_tfl_api(ctx, f'https://api.tfl.gov.uk/StopPoint/Type/{stop_point_type}')
        try:
            r.raise_for_status()
        except:
            logger.exception(f"stop_point_list exception triggered")
            if r.status_code == 404:
                raise ModelRetry("invalid stop point type provided!")
    return [
        {"name": stop_point["commonName"], "naptanId": stop_point["naptanId"]}
        for stop_point in r.json()
        # avoid token length issues - need to perform more intelligent filtering outside of the LLM context window!
        if "camden" in stop_point["commonName"].lower() or 'liverpool' in stop_point["commonName"].lower()
    ]


@tfl_search.tool
async def journey_planner(ctx: RunContext[Deps], from_naptan_id: str, to_naptan_id: str):
    """ Returns potential journeys based on a 'from' stop type naptanId and 'to' stop type naptanId """
    logger.info(f"Retrieving journey plan from naptanId#{from_naptan_id} to naptanId#{to_naptan_id}")
    r = await get_tfl_api(ctx, f'https://api.tfl.gov.uk/Journey/JourneyResults/{from_naptan_id}/to/{to_naptan_id}')
    r.raise_for_status()
    return {leg["instruction"]["summary"] for journey in r.json()["journeys"] for leg in journey["legs"]}


async def main():
    async with AsyncClient(timeout=Timeout(60.0)) as client:
        tfl_api_key = os.getenv("TFL_API_KEY")
        deps = Deps(client=client, tfl_api_key=tfl_api_key)
        result = await tfl_search.run(
            'how can i get from camden town to liverpool street on the underground?', deps=deps
        )
        print('Response:', result.data)


if __name__ == '__main__':
    asyncio.run(main())
