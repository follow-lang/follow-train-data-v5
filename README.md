# Generate train data for follow model

This repo will generate data of `Follow-Lang/set.mm` in huggingface.

## Format

- The data is located in datasets/train.
- Each line is formatted as: Cost(s) Cost(a) Cost(s')\t s a s' .
- Cost(s) = max(Cost(a), 1+Cost(s')).
- The maximum word length is 2048.
- All vocabulary words are listed in datasets/words.txt.
- The data was generated with a depth of 2.

If you need additional data, feel free to reach out.

This version improves readability and flow while maintaining the original meaning.