import { Inter } from "next/font/google";
import { useCallback, useEffect, useState } from "react";
import { SearchResult, Voy } from "voy-search";
import {
    InferenceSession,
    SessionManager,
    AvailableModels,
    initialize,
} from "embd";

const inter = Inter({ subsets: ["latin"] });

const phrases = [
    "That is a very happy Person",
    "That is a Happy Dog",
    "Today is a sunny day",
];

const query = "Is it summer yet?";

async function getSession() {
    await initialize();
    const manager = new SessionManager();
    const loadResult = await manager.loadModel(
        AvailableModels.BAAI_SMALL_EN_v1_5,
        () => console.log("LOADING"),
        (p: number) => console.log("PROGRESS", p)
    );
    return loadResult.unwrapOrElse((err) => {
        console.error(err);
        throw err;
    });
}

async function infer(session: InferenceSession, batch: string[]) {
    const embeddingsResult = await session.run(batch, {});
    //@ts-ignore comlink + errors = bad
    const [state, data] = embeddingsResult.repr;
    if (state === "Ok") {
        return data;
    } else {
        throw new Error(data);
    }
}

function parseEmbeddings(embeddings: Float32Array) {
    let data = [];
    let idx = 0;
    for (let i = 0; i < embeddings.length; i += 384) {
        data.push({
            id: String(idx),
            title: phrases[idx],
            url: `/path/${idx++}`,
            embeddings: Array.from(embeddings.slice(i, i + 384)) as number[],
        });
    }
    return data;
}

export default function Home() {
    const [result, setResult] = useState<SearchResult>();

    const run = useCallback(async (session: InferenceSession) => {
        let start = performance.now();
        const embeddings = await infer(session, phrases);
        const data = parseEmbeddings(embeddings);

        const index = new Voy({ embeddings: data });

        const q = await infer(session, [query]);

        const result = index.search(q, 1);

        setResult(result);
        console.log("Index & Search took", performance.now() - start, "ms");
    }, []);

    useEffect(() => {
        getSession().then(run);
    }, [run]);

    return (
        <main
            className={`flex min-h-screen flex-col justify-center p-4 md:p-24 ${inter.className}`}
        >
            <div className="my-4">
                <h5 className="font-bold">📚 Index:</h5>
                {phrases.map((phrases) => (
                    <p key={phrases}>{phrases}</p>
                ))}
            </div>
            <div className="my-4">
                <h5 className="font-bold">❓ Query: </h5>
                <p>{query}</p>
            </div>
            <div className="my-4">
                <h5 className="font-bold">✨ Search Result</h5>
                {!result && <p>...</p>}
                {result?.neighbors.map((n) => (
                    <p key={n.id}>{n.title}</p>
                ))}
            </div>
        </main>
    );
}
