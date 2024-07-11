
import openai
import pyter

def makeFluid_2hands(my_openai, list_of_words):
  string_of_words = ", ".join(list_of_words) # Converts a list of words into a string with commas: ['bom', 'dia'] into "bom,dia"
  prompt = "Convert into a fluid phrase in portuguese the following: " + string_of_words
  # Now we do our request to ChatGPT
  try:
    completion = my_openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt} ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
  except:
    return "Not able to return a fluid phrase!"

def makeFluid(list_of_words):
  string_of_words = ", ".join(list_of_words) # Converts a list of words into a string with commas: ['bom', 'dia'] into "bom,dia"
  prompt = "Convert into a fluid phrase in portuguese the following: " + string_of_words
  # Now we do our request to ChatGPT
  completion = openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt} ]
  )
  return completion.choices[0].message.content

def getKey():
  key = open("key.txt","r").read().strip('\n')
  return key

def ter(ref, gen):
    return pyter.ter(gen.split(), ref.split())

if __name__ == '__main__':
  #list_of_words = ['cafe', 'chocolate', 'chocolate', 'depois', 'menu']
  list_of_words = ['eu', 'comer', 'acabar', 'depois', 'café', 'tomar']

  #openai.api_key = getKey()

  print(ter("Vou tomar café depois de acabar de comer", "Eu acabo de comer e depois vou tomar café"))
  #print(ter("posso pedir o menu por favor", "menu, por favor"))
  #print(ter("eu gosto muito de chocolate", "eu gosto muito de chocolate"))

  #phrase = makeFluid(list_of_words)
  print(phrase)


  