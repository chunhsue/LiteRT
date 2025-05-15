# Graph-to-graph Transformation
## MHA Optimization
|Dimension|Meaning| Gemma3 1B Example|
|:---:|---|---|
|B| Batch size| 1|
|T| Number of heads| 4|
|N| Sequence length| 128 for prefill / 1 for decode|
|H| Head dimension | 256|

Original multi-head attention in Gemma3 prefill graph
```mermaid
graph TB
  RopeOut --> |"[B, T, N, H]"|Mul["Mul"]
  Mul --> |"[B, T, N, H]"| Transpose["Transpose"]
  Transpose --> |"[B, N, T, H]"| Reshape1["Reshape"]
  Reshape1 --> |"[B, 1, N*T, H]"| MatMul1["MatMul"] & MatMul2["MatMul"]
  MatMul1 --> |"[B, 1, N\*T, T*KV_LEN]"|Concat2
  MatMul2 --> |"[B, 1, N\*T, T]"|Concat2
  Concat2[Concat] --> |"[B, 1, N*T, T\*(KV_LEN+1)]"|Reshape2["Reshape"] 
  Reshape2 --> |"[B, N, T, T\*(KV_LEN+1)]"|Add["Add"]
  Add --> |"[B, N, T, T\*(KV_LEN+1)]"|Reshape3["Reshape"] 
  Reshape3--> |"[B, 1, N*T, T\*(KV_LEN+1)]"|Softmax["Softmax"] 
  Softmax --> |"[B, 1, N*T, T\*(KV_LEN+1)]"|StridedSlice1["StridedSlice"] & StridedSlice2["StridedSlice"]
  StridedSlice1 --> |"[B, 1, N*T, T\*KV_LEN]"|MatMul3["MatMul"]
  StridedSlice2 --> |"[B, 1, N*T, T]"|MatMul4["MatMul"]
  MatMul3 & MatMul4 --> |"[B, 1, N*T, H]"|Add2["Add"]
  Add2--> |"[B, 1, N*T, H]"|Reshape4["Reshape"]
  Reshape4 --> |"[B, T, N, H]"|Transpose1["Transpose"]
  Transpose1 --> |"[B, N, T, H]"|Reshape5["Reshape"]
  Reshape5 --> FCIn

  KCache["K Cache"] --> |"[B, 1, T*KV_LEN, H]"|MatMul1
  KSlice["K Slice"] --> |"[B, T, 1, H]"| ReshapeKSlice[Reshape]
  ReshapeKSlice --> |"[B, 1, T, H]"| MatMul2
  ReshapeKSlice --> |"[B, 1, T, H]"| KSliceOut["K Slice"]
  Mask["Mask"] --> |"[B, 1, T, T\*(KV_LEN+1)]"|Add
  VCache["V Cache"] --> |"[B, 1, H, T*KV_LEN]"|MatMul3
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
  FCIn@{ shape: sm-circ}
  RopeOut@{ shape: sm-circ}
```
After the transformation, the graph will become 
```mermaid
graph TB
  Transpose["Transpose"] --> Reshape1
  Reshape1["Reshape"] --> Split
  Split["Split"] --> SHA1["SHA"] & SHA2["SHA"] & SHA3["SHA"] & SHA4["SHA"] --> Concat2[Concat] --> Reshape2[Reshape]
```
with the following four SHA, based on the number of heads in MHA.
```mermaid
graph TB
  Mul1["Mul"] --> MatMul1["MatMul"] & MatMul2["MatMul"] --> Concat["Concat"] --> Add1["Add"] --> SoftMax["Softmax"] --> StridedSlice1["StridedSlice"] & StridedSlice2["StridedSlice"]
  StridedSlice1 --> MatMul3["MatMul"]
  StridedSlice2 --> MatMul4["MatMul"]
  MatMul3 & MatMul4 --> Add2["Add"]
```