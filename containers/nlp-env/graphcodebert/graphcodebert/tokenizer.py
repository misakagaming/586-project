from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path

paths = [str(x) for x in Path(r"C:\Users\eraye\Desktop\graphcodebert-env\containers\graphcodebert-env\gen2java\cs_translation").glob("*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=50265, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model(r"C:\Users\eraye\Desktop\graphcodebert-env\containers\graphcodebert-env\translation\tokenizer")

paths = [str(x) for x in Path(r"C:\Users\eraye\Desktop\graphcodebert-env\containers\graphcodebert-env\gen2java\cs_translation_segmented").glob("*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=50265, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model(r"C:\Users\eraye\Desktop\graphcodebert-env\containers\graphcodebert-env\translation\tokenizer_segmented")

paths = [str(x) for x in Path(r"C:\Users\eraye\Desktop\graphcodebert_env_new\containers\graphcodebert-env\gen2java\Training-base\train.coolgen")
         .glob("*.gensrc")]

for x in Path(r"C:\Users\eraye\Desktop\graphcodebert_env_new\containers\graphcodebert-env\gen2java\Training-base\train.java").glob("*.java"):
    paths.append(str(x))

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=50265, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model(r"C:\Users\eraye\Desktop\graphcodebert_env_new\containers\graphcodebert-env\translation\tokenizer_coolgen_base")


paths = [str(x) for x in Path(r"C:\Users\eraye\Desktop\graphcodebert_env_new\containers\graphcodebert-env\gen2java\Training2\train.coolgen")
         .glob("*.gensrc")]

for x in Path(r"C:\Users\eraye\Desktop\graphcodebert_env_new\containers\graphcodebert-env\gen2java\Training2\train.java").glob("*.java"):
    paths.append(str(x))

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=50265, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])
#Save the Tokenizer to disk
tokenizer.save_model(r"C:\Users\eraye\Desktop\graphcodebert_env_new\containers\graphcodebert-env\translation\tokenizer_coolgen_mod")

