<div align="center">
  <h1>Voy</h1>
  <strong>A vector similarity search engine in WASM</strong>
</div>

![voy: a vector similarity search engine in WebAssembly][demo]

## Installation

```bash
# with npm
npm i voy

# with Yarn
yarn add voy

# with pnpm
pnpm add voy
```

## APIs

#### `index(input: Resource): SerializedIndex`

**Parameters**

```ts
interface Resource {
  embeddings: Array<{
    id: string; // id of the resource
    title: string; // title of the resource
    url: string; // path to the resource
    embeddings: number[]; // embeddings of the resource
  }>;
}
```

**Return**

```ts
type SerializedIndex = string; // serialized k-d tree
```

#### `search(index: SerializedIndex, query: Query, k: NumberOfResult): Nearests`

**Parameter**

```ts
type SerializedIndex = string; // serialized k-d tree

type Query = number[]; // embeddings of the search query

type NumberOfResult = number; // K top results to return
```

**Return**

```ts
type Nearests = Array<{
  id: string; // id of the nearest resource
  title: string; // title of the nearest resource
  url: string; // path of the nearest resource
  body: string; // body of the nearest resource
  embeddings: number[]; // embeddings of the nearest resource
}>;
```

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

[demo]: ./voy.gif "voy demo"
