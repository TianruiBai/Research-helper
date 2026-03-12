"""PubMed / E-utilities fetcher."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

from src.searcher.base import AbstractFetcher
from src.storage.models import Paper


class PubMedFetcher(AbstractFetcher):
    name = "pubmed"
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    async def search(
        self,
        query: str,
        max_results: int = 200,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[dict]:
        # Step 1: esearch to get PMIDs
        term = query
        if year_start and year_end:
            term += f" AND {year_start}:{year_end}[dp]"
        elif year_start:
            term += f" AND {year_start}:3000[dp]"
        elif year_end:
            term += f" AND 1900:{year_end}[dp]"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await self._request_with_retry(
                client, "GET", self.ESEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": term,
                    "retmax": min(max_results, 1000),
                    "retmode": "json",
                },
            )
            resp.raise_for_status()
            search_data = resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Step 2: efetch to get full records (XML)
            pmids = ",".join(id_list[:max_results])
            resp2 = await self._request_with_retry(
                client, "GET", self.EFETCH_URL,
                params={
                    "db": "pubmed",
                    "id": pmids,
                    "retmode": "xml",
                },
            )
            resp2.raise_for_status()

        return self._parse_xml(resp2.text)

    def _parse_xml(self, xml_text: str) -> list[dict]:
        results: list[dict] = []
        root = ET.fromstring(xml_text)
        for article_el in root.findall(".//PubmedArticle"):
            try:
                medline = article_el.find("MedlineCitation")
                if medline is None:
                    continue
                pmid_el = medline.find("PMID")
                pmid = pmid_el.text if pmid_el is not None else None

                art = medline.find("Article")
                if art is None:
                    continue

                title_el = art.find("ArticleTitle")
                title = title_el.text if title_el is not None else ""

                abstract_el = art.find("Abstract/AbstractText")
                abstract = abstract_el.text if abstract_el is not None else None

                # Authors
                authors: list[str] = []
                for auth in art.findall("AuthorList/Author"):
                    last = auth.findtext("LastName", "")
                    first = auth.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last}, {first}".strip(", "))

                # Year
                year = None
                date_el = art.find("Journal/JournalIssue/PubDate/Year")
                if date_el is not None and date_el.text:
                    try:
                        year = int(date_el.text)
                    except ValueError:
                        pass

                # Journal
                journal_el = art.find("Journal/Title")
                journal = journal_el.text if journal_el is not None else None

                # DOI
                doi = None
                for eid in article_el.findall(".//ArticleId"):
                    if eid.get("IdType") == "doi":
                        doi = eid.text
                        break

                # MeSH terms
                mesh_terms: list[str] = []
                for mh in medline.findall("MeshHeadingList/MeshHeading/DescriptorName"):
                    if mh.text:
                        mesh_terms.append(mh.text)

                results.append({
                    "pmid": pmid,
                    "doi": doi,
                    "title": title or "",
                    "authors": authors,
                    "year": year,
                    "venue": journal,
                    "abstract": abstract,
                    "keywords": mesh_terms,
                })
            except Exception:
                continue
        return results

    def normalise(self, raw: dict) -> Paper:
        doi = raw.get("doi")
        paper_id = Paper.make_id(doi=doi, title=raw.get("title", ""), year=raw.get("year"))

        return Paper(
            id=paper_id,
            doi=doi,
            arxiv_id=None,
            pmid=raw.get("pmid"),
            title=raw.get("title", ""),
            authors=json.dumps(raw.get("authors", [])),
            year=raw.get("year"),
            venue=raw.get("venue"),
            venue_type="journal",
            abstract=raw.get("abstract"),
            keywords=json.dumps(raw.get("keywords", [])),
            citations=None,
            citation_velocity=None,
            influential_citations=None,
            sources=json.dumps(["pubmed"]),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{raw.get('pmid', '')}" if raw.get("pmid") else None,
            fetched_at=datetime.utcnow(),
            is_local=False,
        )
