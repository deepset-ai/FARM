from farm.infer import Inferencer

QA_input = {
    "lang": "en",
    "input": [
        {
            "questions": ["Who did Chopin have at his last Parisian concert in 1848?"],
            "text": "Chopin's public popularity as a virtuoso began to wane, as did the number of his pupils, and this, together with the political strife and instability of the time, caused him to struggle financially. In February 1848, with the cellist Auguste Franchomme, he gave his last Paris concert, which included three movements of the Cello Sonata Op. 65.",
        },
        {
            "questions": ["What instrument did Auguste Franchomme play?"],
            "text": "Chopin's public popularity as a virtuoso began to wane, as did the number of his pupils, and this, together with the political strife and instability of the time, caused him to struggle financially. In February 1848, with the cellist Auguste Franchomme, he gave his last Paris concert, which included three movements of the Cello Sonata Op. 65.",
        },
        {
            "questions": ["What is a regulatory factor produced by macrophages?"],
            "text": "Neutrophils and macrophages are phagocytes that travel throughout the body in pursuit of invading pathogens. Neutrophils are normally found in the bloodstream and are the most abundant type of phagocyte, normally representing 50% to 60% of the total circulating leukocytes. During the acute phase of inflammation, particularly as a result of bacterial infection, neutrophils migrate toward the site of inflammation in a process called chemotaxis, and are usually the first cells to arrive at the scene of infection. Macrophages are versatile cells that reside within tissues and produce a wide array of chemicals including enzymes, complement proteins, and regulatory factors such as interleukin 1. Macrophages also act as scavengers, ridding the body of worn-out cells and other debris, and as antigen-presenting cells that activate the adaptive immune system.",
        },
        {
            "questions": ["What are the most abundant kind of phagocyte?"],
            "text": "Neutrophils and macrophages are phagocytes that travel throughout the body in pursuit of invading pathogens. Neutrophils are normally found in the bloodstream and are the most abundant type of phagocyte, normally representing 50% to 60% of the total circulating leukocytes. During the acute phase of inflammation, particularly as a result of bacterial infection, neutrophils migrate toward the site of inflammation in a process called chemotaxis, and are usually the first cells to arrive at the scene of infection. Macrophages are versatile cells that reside within tissues and produce a wide array of chemicals including enzymes, complement proteins, and regulatory factors such as interleukin 1. Macrophages also act as scavengers, ridding the body of worn-out cells and other debris, and as antigen-presenting cells that activate the adaptive immune system.",
        },
        {
            "questions": ["The Presiding Officer can increase speaking time if what?"],
            "text": "The Presiding Officer (or Deputy Presiding Officer) decides who speaks in chamber debates and the amount of time for which they are allowed to speak. Normally, the Presiding Officer tries to achieve a balance between different viewpoints and political parties when selecting members to speak. Typically, ministers or party leaders open debates, with opening speakers given between 5 and 20 minutes, and succeeding speakers allocated less time. The Presiding Officer can reduce speaking time if a large number of members wish to participate in the debate. Debate is more informal than in some parliamentary systems. Members may call each other directly by name, rather than by constituency or cabinet position, and hand clapping is allowed. Speeches to the chamber are normally delivered in English, but members may use Scots, Gaelic, or any other language with the agreement of the Presiding Officer. The Scottish Parliament has conducted debates in the Gaelic language.",
        },
        {
            "questions": [
                "What does the Presiding Officer try to achieve a balance of between speakers?"
            ],
            "text": "The Presiding Officer (or Deputy Presiding Officer) decides who speaks in chamber debates and the amount of time for which they are allowed to speak. Normally, the Presiding Officer tries to achieve a balance between different viewpoints and political parties when selecting members to speak. Typically, ministers or party leaders open debates, with opening speakers given between 5 and 20 minutes, and succeeding speakers allocated less time. The Presiding Officer can reduce speaking time if a large number of members wish to participate in the debate. Debate is more informal than in some parliamentary systems. Members may call each other directly by name, rather than by constituency or cabinet position, and hand clapping is allowed. Speeches to the chamber are normally delivered in English, but members may use Scots, Gaelic, or any other language with the agreement of the Presiding Officer. The Scottish Parliament has conducted debates in the Gaelic language.",
        },
    ],
}


save_dir = "../saved_models/bert-base-english-squad2"
model = Inferencer(save_dir, gpu=True)
result = model.run_inference(dicts=QA_input["input"])
assert result["predictions"][0]["label"] == "Auguste Franchomme"
