"""JSON request/response models of the API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DescriptorSpec(BaseModel):
    name: str
    metric: Optional[str] = None  # default metric of the descriptor if omitted
    weight: float = 1.0


class SearchRequest(BaseModel):
    image_id: Optional[int] = None  # query by a database image (or a previous result)
    keywords: list[str] = []
    keyword_mode: Literal["and", "or"] = "or"
    descriptors: list[DescriptorSpec] = []
    index: Literal["vptree", "bruteforce"] = "vptree"
    k: int = Field(20, ge=1, le=200)
    # 0-100. Visual: keep results with relative distance <= fuzziness/100 (100 = plain top-K).
    # Keywords: typo tolerance (0 = exact tags only, 100 = most tolerant).
    fuzziness: float = Field(100, ge=0, le=100)


class ResultItem(BaseModel):
    image_id: int
    url: str
    rank: int
    distance: Optional[float] = None
    relative_distance: Optional[float] = None  # 0 = best match, 1 = K-th match (fuzziness filter)
    distances: dict[str, float] = {}
    tags: list[str] = []
    matched_keywords: Optional[int] = None


class QueryInfo(BaseModel):
    kind: Literal["database", "upload", "keywords"]
    image_id: Optional[int] = None
    url: Optional[str] = None
    keywords: list[str] = []
    keyword_mode: str = "or"


class SearchResponse(BaseModel):
    query: QueryInfo
    results: list[ResultItem]
    descriptors: list[DescriptorSpec]
    index_requested: str
    index_used: str
    time_ms: float
    distance_computations: int = 0
    keyword_matches: Optional[int] = None
    keyword_expansions: dict[str, list[str]] = {}  # keyword -> tags actually matched
    fuzziness: float = 100
    k_requested: int = 20
    results_returned: int = 0
    n_images: int
