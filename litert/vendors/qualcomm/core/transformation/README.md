# Graph-to-graph Transformation
## Matmul-convert Fusion
Original graph:
```mermaid
graph TB
  In1 --> MatMul1["MatMul"]
  In2 --> MatMul2["MatMul"]
  MatMul1 --> Convert["Convert"]
  Convert --> |"Quant Params"| Concat["Concat"]
  MatMul2["MatMul"] --> Concat --> Out
  In1@{ shape: sm-circ}
  In2@{ shape: sm-circ}
  Out@{ shape: sm-circ}
```
The updated computational graph after the transformation is shown below.
```mermaid
graph TB
  In1 --> MatMul1["MatMul"]
  In2 --> MatMul2["MatMul"]
  MatMul1 --> |"Quant Params"| Concat["Concat"]
  MatMul2["MatMul"] --> Concat --> Out
  In1@{ shape: sm-circ}
  In2@{ shape: sm-circ}
  Out@{ shape: sm-circ}
```
## Multi-head Attention Optimization
| Dimension	| Description	| Example (Gemma3 1B) |
|:---------:|-------------|---------------------|
| B | Batch size | 1 |
| T | Sequence length | 128 for prefill / 1 for decode |
| N | Number of heads| 4 |
| H | Head dimension | 256 |
* Notation reference: [AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch)

Original MHA (Multi-head Attention) in Gemma3 prefill graph
```mermaid
graph TB
  RopeOut --> |"[B, T, N, H]"| Mul["Mul"]
  Mul --> |"[B, T, N, H]"| Transpose["Transpose"]
  Transpose --> |"[B, N, T, H]"| Reshape1["Reshape"]
  Reshape1 --> |"[B, 1, N*T, H]"| MatMul1["MatMul"] & MatMul2["MatMul"]
  MatMul1 --> |"[B, 1, N\*T, KV_LEN*T]"| Concat2
  MatMul2 --> |"[B, 1, N\*T, 1*T]"| Concat2
  Concat2[Concat] --> |"[B, 1, N*T, (KV_LEN+1)\*T]"| Reshape2["Reshape"] 
  Reshape2 --> |"[B, N, T, (KV_LEN+1)\*T]"| Add["Add"]
  Mask["Mask"] --> |"[B, 1, T, (KV_LEN+1)\*T]"| Add
  Add --> |"[B, N, T, (KV_LEN+1)\*T]"| Reshape3["Reshape"] 
  Reshape3--> |"[B, 1, N*T, (KV_LEN+1)\*T]"| Softmax["Softmax"] 
  Softmax --> |"[B, 1, N*T, (KV_LEN+1)\*T]"| StridedSlice1["StridedSlice"] & StridedSlice2["StridedSlice"]
  StridedSlice1 --> |"[B, 1, N*T, KV_LEN\*T]"| MatMul3["MatMul"]
  StridedSlice2 --> |"[B, 1, N*T, T]"| MatMul4["MatMul"]
  MatMul3 & MatMul4 --> |"[B, 1, N*T, H]"| Add2["Add"]
  Add2--> |"[B, 1, N*T, H]"| Reshape4["Reshape"]
  Reshape4 --> |"[B, N, T, H]"| Transpose1["Transpose"]
  Transpose1 --> |"[B, T, N, H]"| Reshape5["Reshape"]
  Reshape5 --> |"[B, T, N*H]"| FCIn

  KCache["K Cache"] --> |"[B, 1, KV_LEN*T, H]"| MatMul1
  KSlice["K Slice"] --> |"[B, T, 1, H]"| ReshapeKSlice[Reshape]
  ReshapeKSlice --> |"[B, 1, 1*T, H]"| MatMul2
  ReshapeKSlice --> |"[B, 1, T, H]"| KSliceOut["K Slice"]

  VCache["V Cache"] --> |"[B, 1, H, KV_LEN*T]"| MatMul3
  VSlice["V Slice"] --> |"[B, T, 1, H]"| TransposeVSlice[Transpose]
  TransposeVSlice --> |"[B, 1, H, T]"| MatMul4
  TransposeVSlice --> |"[B, 1, H, T]"| VSliceOut["V Slice"]

  KCache@{shape: text}
  KSlice@{shape: text}
  KSliceOut@{shape: text}
  Mask@{shape: text}
  VCache@{shape: text}
  VSlice@{shape: text}
  VSliceOut@{shape: text}
  FCIn@{shape: sm-circ}
  RopeOut@{shape: sm-circ}
```
The updated computational graph after the transformation is shown below.
```mermaid
graph TB
  RopeOut --> |"[B, T, N, H]"| Transpose["Transpose"]
  Transpose --> |"[B, N, T, H]"| Reshape1
  Reshape1["Reshape"] --> |"[B, 1, N*T, H]"| Split
  Split["Split
  (N=4)"] --> |"[B, 1, T, H]"| SHA1["SHA"] & SHA2["SHA"] & SHA3["SHA"] & SHA4["SHA"]
  SHA1 & SHA2 & SHA3 & SHA4 --> |"[B, 1, T, H]"| Concat2["Concat 
  (N=4)"]
  Concat2 --> |"[B, 1, T, N*H]"| Reshape2[Reshape]
  Reshape2 --> |"[B, T, N*H]"| FCIn
  SHAInputs["SHA Inputs"] --> SHA1 & SHA2 & SHA3 & SHA4
  KSlice["K Slice"] --> ReshapeKSlice[Reshape]
  ReshapeKSlice --> SHAInputs
  ReshapeKSlice --> KSliceOut["K Slice"]
  VSlice["V Slice"] --> TransposeVSlice[Transpose]
  TransposeVSlice --> VSliceOut["V Slice"]
  TransposeVSlice --> SHAInputs

  KCache["K Cache"] --> SHAInputs
  Mask["Mask"] --> SHAInputs
  VCache["V Cache"] --> SHAInputs

  KCache@{shape: text}
  KSlice@{shape: text}
  Mask@{shape: text}
  VCache@{shape: text}

  SHAInputs@{shape: text}
  KSliceOut@{shape: text}
  VSliceOut@{shape: text}
  KSlice@{shape: text}
  VSlice@{shape: text}
  FCIn@{shape: sm-circ}
  RopeOut@{shape: sm-circ}
```
with the following four SHA (Single-head Attention), based on the number of heads in MHA.
```mermaid
graph TB
  SHAIn --> |"[B, 1, T, H]"| Mul1
  Mul1["Mul"] --> |"[B, 1, T, H]"| MatMul1["MatMul"] & MatMul2["MatMul"]
  MatMul1 --> |"[B, 1, T, KV_LEN*T]"| Concat["Concat"]
  MatMul2 --> |"[B, 1, T, T]"| Concat["Concat"]
  Concat --> |"[B, 1, T, (KV_LEN+1)\*T]"| Add["Add"]
  Add --> |"[B, 1, T, (KV_LEN+1)\*T]"| SoftMax["Softmax"] 
  SoftMax --> |"[B, 1, T, (KV_LEN+1)\*T]"| StridedSlice1["StridedSlice"] & StridedSlice2["StridedSlice"]
  StridedSlice1 --> |"[B, 1, T, KV_LEN\*T]"| MatMul3["MatMul"]
  StridedSlice2 --> |"[B, 1, T, T]"| MatMul4["MatMul"]
  KCache["K Cache"] --> |"[B, 1, KV_LEN*T, H]"| MatMul1
  KSlice["K Slice"] --> |"[B, T, 1, H]"| ReshapeKSlice[Reshape]
  ReshapeKSlice --> |"[B, 1, T, H]"| MatMul2
  ReshapeKSlice --> |"[B, 1, T, H]"| KSliceOut["K Slice"]
  Mask["Mask"] --> |"[B, 1, T, (KV_LEN+1)\*T]"| Add
  VCache["V Cache"] --> |"[B, 1, H, KV_LEN*T]"| MatMul3
  VSlice["V Slice"] --> |"[B, T, 1, H]"| TransposeVSlice[Transpose]
  TransposeVSlice --> |"[B, 1, H, T]"| MatMul4
  TransposeVSlice --> |"[B, 1, H, T]"| VSliceOut["V Slice"]
  MatMul3 & MatMul4 --> |"[B, 1, T, H]"| Add2["Add"]
  Add2 --> |"[B, 1, T, H]"| SHAOut
  SHAIn@{shape: sm-circ}
  SHAOut@{shape: sm-circ}
  KCache@{shape: text}
  Mask@{shape: text}
  VCache@{shape: text}
  KSlice@{shape: text}
  VSlice@{shape: text}
  KSliceOut@{shape: text}
  VSliceOut@{shape: text}
```
