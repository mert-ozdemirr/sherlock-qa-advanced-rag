from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

def recursive_chunker(chapters_dict, chunk_size_char):
    all_chunks_dict_list = []

    recursive_chapter_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_char, chunk_overlap=0, separators=[r"\n\n", r"(?<=[.!?])\s+(?=[A-Z])"], is_separator_regex=True)

    for idx, chapter_dict in enumerate(chapters_dict):
        split_chapter_content = recursive_chapter_splitter.split_text(chapter_dict["chapter_content"])
        for jdx, chunk_text in enumerate(split_chapter_content):
            chunk_dict = {
                "book_name": chapter_dict["book_name"],
                "part_name": chapter_dict["part_name"],
                "chapter_name": chapter_dict["chapter_name"],
                "chunk_number": jdx+1,
                "chunk_content": chunk_text
            }
            all_chunks_dict_list.append(chunk_dict)
    
    return all_chunks_dict_list

def recursive_chunking_pkl_to_pkl(chapters_list_file_path, chunks_list_file_path_output, chunk_size_char):
    with open(chapters_list_file_path, "rb") as f:
        chapters_as_dict = pickle.load(f)
    
    output_list = recursive_chunker(chapters_as_dict, chunk_size_char)

    with open(chunks_list_file_path_output, "wb") as f:
        pickle.dump(output_list, f)

#recursive_chunking_pkl_to_pkl("data/all_novels_chapters_w_metadata_list.pkl", "data/all_chunks_w_metadata_list.pkl", 450)
        
"""with open("data/all_chunks_w_metadata_list.pkl", "rb") as f:
        chunks_as_dict = pickle.load(f)
print(len(chunks_as_dict))
for i, d in enumerate(chunks_as_dict):
    if i == 1000:
        print(d)
        break"""

"""with open("data/all_chunks_w_metadata_list.pkl", "rb") as f:
        chunks_as_dict = pickle.load(f)
with open("data/450_recursivechunks_w_metadata.txt", "w") as f:
     for i in chunks_as_dict:
          f.write(str(i))
          f.write("\n---\n")"""