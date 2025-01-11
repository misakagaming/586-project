with open(f"test_1.output",encoding="utf-8") as f2:
    outputs = f2.readlines()
    splitter = outputs[0][1]
    for idx in range(len(outputs)):
        code=outputs[idx].split(splitter)[1].strip("\n")
        print(code)