from collections import defaultdict, deque

def get_input_data():
    with open('data/AlienAlphabet.txt', encoding='utf-8') as f:
        data = f.read()
    return data


def find_character_order(words):
    # Build adjacency graph and in-degree count
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    all_chars = set()

    # Collect all characters
    for word in words:
        for char in word:
            all_chars.add(char)
            if char not in in_degree:
                in_degree[char] = 0

    # Compare consecutive words to determine ordering
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))

        # Find first differing character
        for j in range(min_len):
            if word1[j] != word2[j]:
                # word1[j] comes before word2[j]
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break

    # Topological sort using Kahn's algorithm
    queue = deque([char for char in all_chars if in_degree[char] == 0])
    result = []

    while queue:
        char = queue.popleft()
        result.append(char)

        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if valid ordering exists
    if len(result) != len(all_chars):
        return None  # Cycle detected

    return ''.join(result)


if __name__ == "__main__":
    words = get_input_data().strip().split('\n')
    order = find_character_order(words)

    # Map to English alphabet
    english_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alien_to_english = {alien_char: english_alphabet[i] for i, alien_char in enumerate(order)}

    # Translate words to English
    translated_words = []
    for word in words:
        translated = ''.join(alien_to_english.get(char, char) for char in word)
        translated_words.append(translated)

    # Write to file
    with open('character_order.txt', 'w', encoding='utf-8') as f:
        f.write(f"Character order: {order}\n")
        f.write(f"Number of characters: {len(order)}\n\n")
        f.write("Alien to English mapping:\n")
        for alien_char, english_char in alien_to_english.items():
            f.write(f"{alien_char} -> {english_char}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("Input text translated to English:\n")
        f.write("="*50 + "\n")
        for i, (alien_word, english_word) in enumerate(zip(words, translated_words), 1):
            f.write(f"{i:3}. {alien_word:15} -> {english_word}\n")

    # Also print
    try:
        print(f"Character order: {order}")
        print(f"Number of characters: {len(order)}")
        print("\nAlien to English mapping:")
        for alien_char, english_char in alien_to_english.items():
            print(f"{alien_char} -> {english_char}")
        print("\n" + "="*50)
        print("Input text translated to English:")
        print("="*50)
        for i, (alien_word, english_word) in enumerate(zip(words, translated_words), 1):
            print(f"{i:3}. {alien_word:15} -> {english_word}")
    except UnicodeEncodeError:
        print(f"Character order written to character_order.txt")
        print(f"Number of characters: {len(order)}")