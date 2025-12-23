import re

def hyperparameter_opt_test_data_setup(synthetic_data_txt_path):
    with open(synthetic_data_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks by TEST ITEM markers
    raw_items = re.split(r"=== TEST ITEM\s+(\d+)\s+===", text)
    # raw_items looks like ['', '1', '...item1...', '2', '...item2...', ...]

    test_items = []

    # iterate over pairs (id, content)
    for i in range(1, len(raw_items), 2):
        item_id = int(raw_items[i])
        item_text = raw_items[i+1]

        # Extract QUERY block
        query = re.search(
            r"QUERY:\s*(.+?)(?=\nREFERENCE:)", 
            item_text, 
            flags=re.DOTALL
        )
        query = query.group(1).strip() if query else ""

        # Extract REFERENCE block
        reference = re.search(
            r"REFERENCE:\s*(.+?)(?=\nSCENARIO:)",
            item_text,
            flags=re.DOTALL
        )
        reference = reference.group(1).strip() if reference else ""

        # Extract SCENARIO block
        scenario = re.search(
            r"SCENARIO:\s*(.+?)(?=\n===|$)",
            item_text,
            flags=re.DOTALL
        )
        scenario = scenario.group(1).strip() if scenario else ""

        scenario_lines = scenario.splitlines()
        scenario = ".\n".join(scenario_lines[:-2])

        test_items.append({
            "id": item_id,
            "query": query,
            "reference": reference,
            "scenario": scenario
        })

    return test_items

#items = hyperparameter_opt_test_data_setup("data/ragas_synthetic_whole_corpus.txt")
