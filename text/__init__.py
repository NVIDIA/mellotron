""" from https://github.com/keithito/tacotron """
import re
import random
from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
_words_re = re.compile(r"([a-zA-ZÀ-ž]+['][a-zA-ZÀ-ž]{1,2}|[a-zA-ZÀ-ž]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž{}]+)")


def get_arpabet(word, dictionary):
  word_arpabet = dictionary.lookup(word)
  if word_arpabet is not None:
    return "{" + word_arpabet[0] + "}"
  else:
    return word


def text_to_sequence(text, cleaner_names, dictionary=None, p_arpabet=1.0):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      clean_text = _clean_text(text, cleaner_names)
      if cmudict is not None:
        words = _words_re.findall(text)
        clean_text = [
          get_arpabet(word[0], dictionary)
          if ((word[0] != '') and random.random() < p_arpabet) else word[1]
          for word in words]

        for i in range(len(clean_text)):
            t = clean_text[i]
            sequence += _arpabet_to_sequence(t[1:-1])
            #sequence += space
      else:
        sequence += _symbols_to_sequence(clean_text)
      break
    
    else:
      sequence += text_to_sequence(m.group(1), cleaner_names, dictionary, p_arpabet)
      sequence += _arpabet_to_sequence(m.group(2))
      text = m.group(3)

  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

