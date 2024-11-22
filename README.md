![Banner](https://github.com/nih23/sisax-hackathon/blob/main/assets/ai-insights-header-1000x505.jpg?raw=true)


# SiSax Hackathon 👩‍💻
Der Hackathon besteht aus zwei Richtungen: 
1. Analyse von Blockdrucken und Handschriften
2. KI am Tisch: Skat trifft Machine Learning!

Gemeinsam mit der [@StabiBerlin](https://github.com/StabiBerlin) wollen wir **historische Dokumente mit KI** erkunden! In diesem Hackathon kombinieren wir modernste Technologien mit der Analyse jahrhundertealter **tibetischer Handschriften und Drucke**. Entwickle eine KI, die historische Blockdrucke und Manuskripte analysiert, Layouts erkennt und Texte vergleicht. Die Staatsbibliothek zu Berlin stellt einige historische Objekte im Rahmen einer Ausstellung zur Verfügung! Die Teilnehmenden des Hackathons haben exklusiven Zugang zu diesen wertvollen Originaldokumenten und können selbst Bilder erstellen und die Objekte aus nächster Nähe betrachten. So lässt sich die Arbeit mit den digitalen Daten ideal mit den physischen Artefakten kombinieren.

Beim **Skat-Hackathon** entwickelst du in einem Team eine KI, die in der Lage ist, Skat-Karten auf der Hand und auf dem Tisch zu erkennen – und das alles in Echtzeit! Mit Hilfe modernster Objekterkennungsmodelle wie YOLO, SAM2 und GroundingDino erfasst deine KI die Spielkarten und entscheidet, welche Karte als nächstes gespielt werden soll. Mithilfe von leistungsstarken Large Language Models (LLMs), darunter selbst gehostete Open-Source-Modelle, wird deine KI auch die optimale Spielstrategie entwickeln.

Am Ende des Hackathons treten die besten KI-Systeme gegeneinander an – und du hast die Chance auf tolle Preise!

Weitere Details zum Hackathon findest du auf unserer [Webseite](https://ki-dresden.net/ai-insights-saxony-hackathon).



# Getting started
Das Python Paket kann mit folgendem Befehl installiert werden:
```bash
git clone https://github.com/nih23/sisax-hackathon
cd sisax-hackathon
pip install -e .
```

Im `demo_notebook.ipynv` und `demo.py` wird Objekterkennung und Interaktion mit einem selbst-gehosteten LLM mit dem sisax Paket. Es ist zu beachten, dass die entsprechende Keys, Hosts, Ports, etc. in der .env zu spezifizieren sind. 

# ML Ressourcen
Ihr müsst in der `.env` noch die bei [@cloudandheat](https://github.com/cloudandheat) gehosteten Modelle ergänzen, damit der aktuelle Code aus dem Paket verwendet und z.B. `demo.py` ausgeführt werden kann.

```bash
OBJECT_DETECTION_HOST=http://185.128.117.252:8000
OLLAMA_HOST=http://...
OLLAMA_PORT=11434
OPENAI_API_KEY=sk-proj-..
```

Wir stellen zwei selbst-gehostete Compute Nodes mit LLMs zur Verfügung. Standard-LLMs können von `185.128.119.247` benutzt werden. Hierzu zählen:

| Model              | Parameters | Size  | model (for API request)                  |
| ------------------ | ---------- | ----- | -------------------------------- |
| Llama 3.2          | 3B         | 2.0GB | llama3.2     |
| Llama 3.2          | 1B         | 1.3GB | llama3.2:1b     |
| Llama 3.1          | 8B         | 4.7GB | llama3.1     |
| Phi 3 Mini         | 3.8B       | 2.3GB | phi3     |
| Phi 3 Medium       | 14B        | 7.9GB | phi3:medium     |
| Gemma 2            | 2B         | 1.6GB | gemma2:2b     |
| Gemma 2            | 9B         | 5.5GB | gemma2     |
| Mistral            | 7B         | 4.1GB | mistral     |
| Moondream 2        | 1.4B       | 829MB | moondream     |
| Neural Chat        | 7B         | 4.1GB | neural-chat     |
| Starling           | 7B         | 4.1GB | starling-lm     |
| LLaVA              | 7B         | 4.5GB | llava     |
| Solar              | 10.7B      | 6.1GB | solar     |

Ein Big-LLM stellen wir über folgenden Host zur Verfügung: `185.128.117.91`. 

| Model              | Parameters | Size  | model (for API request)                  |
| ------------------ | ---------- | ----- | -------------------------------- |
| Llama 3.1          | 70B         | ... | llama3.1:70b     |
| Llama 3.2 Vision   | 90B         | ... | llama3.2-vision:90b     |


Die OpenAI API Keys werden zu Beginn des Hackathons mitgeteilt. 

# Next steps
Ihr könnt das vorliegende Python Paket um weitere Funktionalitän erweitern. Beispielsweise sind folgendes **Features** sinnvoll:
- simples WebUI mit [Flask](https://github.com/pallets/flask) oder [gradio](https://github.com/gradio-app/gradio)
- Auslesen der Webcam per [opencv](https://github.com/opencv/opencv-python)

# Mögliche Bibliotheken und Tools
Die nachfolge Auflisting ist eher als Inspiration gedacht. Es gibt eine große Menge an open-source Tools für verschiedene Aufgaben. Teils gibt es schon Ansätze wie lang-segment-anything oder GPT4o, die ohne fine-tuning (d.h. Training auf GPU) zur Erkennung von Spielkarten verwendet werden können. Fine-tuning von YoloV11 mit `Ultralytics` oder dem neuesten Ansatz `GLEE` könnte jedoch eine deutlich höhere Genauigkeit versprechen.

- Layout-Analyse
  - [unstructured](https://github.com/Unstructured-IO/unstructured)
  - [LayoutReader](https://github.com/ppaanngggg/layoutreader?tab=readme-ov-file)
- Objekterkennung
  - [ultralytics](https://github.com/ultralytics) (fine-tuning notwendig)
  - [Glee](https://github.com/FoundationVision/GLEE) (fine-tuning notwendig)
  - [Lang-Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)  
