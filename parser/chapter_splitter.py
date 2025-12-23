from pathlib import Path
import re
import pickle as pkl

def load_book(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


PART_PATTERN = re.compile(r"^PART\s+[IVXLC]+\.?$", re.IGNORECASE)


def split_into_parts(text: str) -> list[str]:
    parts_text = re.split(r"(?=PART II)", text)

    parts = []

    for part in parts_text:
        part_splitted = part.splitlines()
        part_name = "\n".join(part_splitted[:2])
        part_content = "\n".join(part_splitted[2:])
        part_dict = {
            "part_name" : part_name,
            "part_content" : part_content
        }

        parts.append(part_dict)

    return parts


def split_into_chapters(text):
    chapter_splits = re.split(r"(?i)(?=chapter\s+(?:[ivxlcdm]+|\d+)\b)", text)

    chapters = []

    for chapter in chapter_splits:
        chapter_splitted = chapter.splitlines()
        chapter_name = "\n".join(chapter_splitted[:2])
        chapter_content = "\n".join(chapter_splitted[2:])
        chapter_dict = {
            "chapter_name" : chapter_name,
            "chapter_content" : chapter_content
        }

        chapters.append(chapter_dict)
    
    return chapters

def parse_book(book_path):
    book_parse_storage = []

    book_text = load_book(book_path)
    book_parts = split_into_parts(book_text)

    # book has parts
    if len(book_parts) > 1:
        for book_part in book_parts:
            part_list = []
            chapters = split_into_chapters(book_part["part_content"])
            for i, chapter in enumerate(chapters):
                if i != 0:
                    chapter["part_name"] = book_part["part_name"]
                    part_list.append(chapter)
            book_parse_storage.append(part_list)
    # book does not have any parts
    else:
        one_list = []
        chapters = split_into_chapters(book_text)
        for i, chapter in enumerate(chapters):
            if i != 0:
                chapter["part_name"] = "NA"
                one_list.append(chapter)
        book_parse_storage.append(one_list)

    return book_parse_storage


book1_path = "/Users/mertozdemir/Documents/Tech Narts/Orientation Training/Sherlock-Project/data/novels-raw/txt/1_a_study_in_scarlet.txt"
book2_path = "/Users/mertozdemir/Documents/Tech Narts/Orientation Training/Sherlock-Project/data/novels-raw/txt/2_the_sgin_of_four.txt"
book3_path = "/Users/mertozdemir/Documents/Tech Narts/Orientation Training/Sherlock-Project/data/novels-raw/txt/3_the_hound_of_the_baskervilles.txt"
book4_path = "/Users/mertozdemir/Documents/Tech Narts/Orientation Training/Sherlock-Project/data/novels-raw/txt/4_the_valley_of_fear.txt"

def list_single_book(book_path, book_name):
    all_chapters = []
    for idx, dict_list in enumerate(parse_book(book_path)):
        for d in dict_list:
            d["book_name"] = book_name
            d["part_name"] = d["part_name"].lstrip("\ufeff")
            d["chapter_name"] = d["chapter_name"].lstrip("\ufeff")
            all_chapters.append(d)
    return all_chapters

def all_books_list_write(book_path_name_tuplist, write_path):
    all_chapters_4_books = []
    for book_tup in book_path_name_tuplist:
        all_chapters_4_books.extend(list_single_book(book_tup[0], book_tup[1]))
    
    with open(write_path, "wb") as f:
        pkl.dump(all_chapters_4_books, f)



"""books = []
books.append((book1_path, "A Study In Scarlet"))
books.append((book2_path, "The Sign of Four"))
books.append((book3_path, "The Hound of the Baskervilles"))
books.append((book4_path, "The Valley of Fear"))

all_books_list_write(books, "data/all_novels_chapters_w_metadata_list.pkl")"""

    

    