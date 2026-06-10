"""Validate that mirage-ai's GitHub resource can read repo files programmatically."""
import asyncio
import os


async def main():
    from mirage import MountMode, Workspace
    from mirage.resource.github import GitHubConfig, GitHubResource

    token = os.environ["GITHUB_TOKEN"]

    # public repo
    pub = GitHubResource(config=GitHubConfig(token=token),
                         owner="strukto-ai", repo="mirage", ref="main")
    ws = Workspace({"/gh": pub}, mode=MountMode.READ)
    out = await (await ws.execute("cat /gh/README.md")).stdout_str()
    print("PUBLIC README line1:", out.splitlines()[0][:70])

    # our private OntoRAG dataset repo
    priv = GitHubResource(config=GitHubConfig(token=token),
                          owner="openfantasymap", repo="amol-ontorag", ref="main")
    ws2 = Workspace({"/d": priv}, mode=MountMode.READ)
    man = await (await ws2.execute("cat /d/manifest.json")).stdout_str()
    import json
    m = json.loads(man)
    print("PRIVATE manifest ok: v%s, %d entities, %d chunks" % (
        m["dataset"]["version"], m["ontology"]["counts"]["entities"],
        m["content"]["counts"]["chunks"]))
    ls = await (await ws2.execute("find /d/content/chunks -name '*.jsonl'")).stdout_str()
    print("chunk files via find:", len(ls.split()), "->", ls.split()[:2])
    print("MIRAGE_OK")


asyncio.run(main())
