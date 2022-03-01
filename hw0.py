def gen_sentences(path):
    with open(path, encoding="utf8") as file:
        for line in file:
            data=line.strip("\n")
            if len(data)!=0:
                yield data.split(" ")


def case_sarcastically(text):
    result=""
    count=0
    for i in text:
        if i.upper()!=i.lower():
            if count%2==0:
                result+=i.lower()
            else:
                result+=i.upper()
            count+=1
        else:
            result+=i
    return result

def prefix(s, n):
    if len(s)<n:
        raise ValueError(f"Value {n} for n is larger than the length of s")
    elif n<=0:
        raise ValueError(f"Value {n} for n is negative")
    return s[:n]


def suffix(s, n):
    if len(s)<n:
        raise ValueError(f"Value {n} for n is larger than the length of s")
    elif n<=0:
        raise ValueError(f"Value {n} for n is negative")
    return s[len(s)-n:]


def sorted_chars(s):
    result=[]
    for i in s:
        result.append(i)
    result.sort()
    data=[]
    for i in result:
        if i in data:
            continue
        data.append(i)
    return data
