"""Pydantic result schemas for multi-model SerpAPI orchestration."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    position: int | None = None
    title: str
    link: str
    snippet: str | None = None
    source: str | None = None
    date: str | None = None


class LocalResult(BaseModel):
    title: str
    address: str
    phone: str | None = None
    rating: float | None = None
    reviews: int | None = None
    place_id: str | None = None
    website: str | None = None


class ShoppingResult(BaseModel):
    title: str
    price: str
    source: str
    link: str
    rating: float | None = None
    reviews: int | None = None
    shipping: str | None = None


class NewsResult(BaseModel):
    title: str
    link: str
    source: str
    date: str | None = None
    snippet: str | None = None


class ScholarResult(BaseModel):
    title: str
    link: str
    authors: list[str] | None = None
    publication: str | None = None
    year: int | None = None
    cited_by: int | None = None
    snippet: str | None = None


class ImageResult(BaseModel):
    title: str
    link: str
    original: str
    thumbnail: str
    source: str
    width: int | None = None
    height: int | None = None
