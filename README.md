
![Logo](https://i.imgur.com/x2BH3WB.jpeg)



[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![Language](https://img.shields.io/github/languages/top/EricTelekom/fortneit)](https://www.google.de)

#
Projekt zur Bilderkennung und Klassifizierung von Bildern, Videos und eines Video-Feeds.



## Demo

![Demo](demo.gif)

Features:
------------------------
- Objekterkennung für Bilder und Videos ([siehe COCO Dataset](#datenquellen))
- Live-Objekterkennung für Livestreams und die Webcam
- Speichern der Ergebnisse
- dedizierte und exakte Ausgabe der Ergebnisse / klassifizierten Objekte


Todo's:
------------------------
- [ ] Modelltraining Objekterkennung initialisieren
- [ ] Live Video Object Detection (LVOD) als standalone Webanwendung bauen 
- [ ] LVOD effizienter & performanter machen (siehe Framerate)



Quellen:
------------------------
- Allgemeines zu Künstliche Intelligenz: https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz
- OpenAPI (maschinelles Lernen): https://beta.openai.com/overview
- EleutherAI (Künstliche Intelligenz): https://6b.eleuther.ai/
- Testim (Automation von Sachverhalten): https://app.testim.io/#/signin
- Allgemeines zu Objekterkennung: https://www.fritz.ai/object-detection/
- Beispielprojekt (ObjektDetektion): https://github.com/fizyr/keras-retinanet
- Beispielprojekt (ImageAI): https://github.com/OlafenwaMoses/ImageAI/tree/master/test
- ImageNet: VGGNet, ResNet, Inception, and Xception mit Keras: https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
- ImageAI Objekterkennung: https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606
- ImageAI Objekterkennung Training: https://medium.com/deepquestai/train-object-detection-ai-with-6-lines-of-code-6d087063f6ff
- ImageAI Google Collab: https://colab.research.google.com/drive/1R6t5MfFc3JnhZB-UmTjWLbEetBZ22agg#scrollTo=ElREAYhfHVya
- Alibaba DSL Communication / Dataset utilization: https://www.alibabacloud.com/blog/how-do-you-use-deep-learning-to-identify-ui-components_597859
- Apple "Barrierefreiheit durch maschinelles Lernen": https://machinelearning.apple.com/research/mobile-applications-accessible
- Darknet YOLO ("You Only Look Once") Algorithmus & System: https://pjreddie.com/darknet/yolo/
- Datasets über neurale Netzwerke generieren & trainieren: https://www.youtube.com/watch?v=Rgpfk6eYxJA
- Algorithmen im Vergleich: https://jonathan-hui.medium.com/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359
- Tensorflow Objektdetektion: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
- Code Grepper Code Database: https://www.codegrepper.com/code-examples/

- Tensorflow Objekterkennung: https://www.youtube.com/watch?v=yqkISICHH-U
- Tensorflow 2 Model Zoo (trainierte 2017 COCO Modelle): https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
- Tensorflow Gesamtmodell: https://github.com/tensorflow/models
- Tensorflow getestete Build-Konfigurationen (CUDA / cuDNN): https://www.tensorflow.org/install/source_windows

- Jupyter Notebook - Lokale Anwendung zur visualisierung des Codes bezüglich Bildaufnahme und Modelltraining: https://jupyter.org/

# Datenquellen:
------------------------
- Open Images Dataset: https://storage.googleapis.com/openimages/web/index.html
- OSF: https://osf.io/ezg7j/
- FACES: https://faces.mpdl.mpg.de/imeji/collection/IXTdg721TwZwyZ8e?q=
- KDEF: https://www.emotionlab.se/kdef/
- MACBrain Resources: https://macbrain.org/resources/
- COCO Dataset: https://cocodataset.org/#explore

Anwendungsbeispiele:
------------------------
- https://www.topbots.com/chihuahua-muffin-searching-best-computer-vision-api/
- C3




# Yolov5 Objekterkennung und Modelltraining
------------------------

Features:
------------------------
- Objekterkennung, welche das .pt Dateiformat nutzt (Pytorch)
- Modelltraining, welche .yaml-Dateien nutzt
- farbliche Abstimmung der Erkennung (rot=+15%, grün=+70%)
------------------------

Objekterkennung Konfigurationen
- mit einer Eingabeaufforderung, welche Zugriff auf Python hat in den "yolov5" Ordner steuern
- dort mit dem Befehl "python detect.py --weights Modell.pt --img 640 --conf 0.15 --source data/images" das Programm starten
- Argumente um die Erkennung zu steuern:
    - "--weights Modell.pt" Name des Modells, welches genutzt werden soll (im selben Verzeichnis)
    - "--img 640" Skalierung des genutzen Bildes (weniger Auflösung = schnellere und schlechtere Erkennungen)
    - "--conf 0.15" Mindestwert, welcher erreicht werden muss, bevor einer Erkennung deklariert wird
    - "--source 0" Input Quelle (Ordner, 0 für Webcam, screen 0 für Bildschirmaufnahme, Webadressen)
    - "--device" Grafikkarte (0,1,2), welche genutzt werden soll, oder "CPU"
    - "--view-img False" blendet Live-Ergebnis aus
    - "--save-txt True" speichert Ergebnisse (Bounding Boxes) als .txt-Dateien
    - "--save-conf True" speichert Erkennungswahrscheinlichkeiten in der selben Dateien
    - "--nosave True" speichert nichts
    - "--classes" Filtert nach Klassen (0,4,8)
    - "--name" Name, unter dem der Testlauf gespeichert wird
    - "--line-thickness" Breite der Bounding Box in Pixeln
    - "--hide-labels True" Label werden in der Live-Vorschau nicht angezeigt
    - "--hide-conf True" Wahrscheinlichkeit wird in der Live-Vorschau nicht angezeigt
------------------------

Für das Modelltraining wird die selbe requirements-Datei genutzt, dabei ist zu beachten, dass auf der Trainingsmaschine CUDA und cuDNN (NVIDIA) installiert sein müssen, um effizientes Training über die Grafikkarte zu garantieren.

Verzeichnisformatierung (Training Gesichtserkennung):<br>
|-- yolov5 <br>
&nbsp;&nbsp;&nbsp;&nbsp;|-- data <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- gesicht.yaml <br>
&nbsp;&nbsp;&nbsp;&nbsp;|-- datasets <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- images <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 01.jpg <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 02.jpg <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- etc... <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- labels <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 01.txt <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 02.txt <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- etc... <br>
&nbsp;&nbsp;&nbsp;&nbsp;|-- runs <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- train <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- exp01 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- result.pt <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- detect <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- exp01 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 1.jpg <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- 2.jpg <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- etc... <br>
&nbsp;&nbsp;&nbsp;&nbsp;|-- gesicht.pt <br>
&nbsp;&nbsp;&nbsp;&nbsp;|-- detect.py <br>
        
------------------------

## YOLOv5 Objekterkennung Glossar

|Begriff|Erklärung|
|---|---|
|.engine|[Modellformat TensorRT](https://developer.nvidia.com/tensorrt)|
|.mlmodel|[Modellformat CoreML](https://developer.apple.com/documentation/coreml)
|.onnx|[Modellformat ONNX](https://onnx.ai/)|
|.pb|[Modellformat TensorFlow GraphDef](https://www.tensorflow.org/api_docs/python/tf/compat/v1/GraphDef) (outdated)|
|.pt|[Modellformat Pytorch](https://pytorch.org/)|
|.tflite|[Modellformat TensorFlow Lite](https://www.tensorflow.org/lite) (mobil)|
|.torchscript|[Modellformat Torchscript](https://pytorch.org/docs/stable/jit.html)|
|.yaml|Dateiformat für die Modellkonfigurationsdatei im Training|
|Anchor boxes|Vordefinierte bounding boxen, wird als Startpunkt für den Algorithmus genutzt|
|Augmented interference|Bild wird mehrmals genutzt, aber nie gleich gezeigt ([„Mosaik-Bildung“](https://docs.ultralytics.com/FAQ/augmentation/))|
|Autoanchor|Automatisches generieren von „anchor Boxen“|
|batch|Datengruppierung, welche zusammen verarbeitet wird, ohne eindeutige Zusammenhänge zu haben (außer die Klasse)|
|Batch normalization|SyncBatchNorm, genutzt, um mehrere GPUs parallel trainieren zu lassen, normalisiert Statistiken mehrerer Devices auf einheitliche Werte|
|Batch-size|Größe der Datengruppierung (Binär), -1 für autobatch|
|Bounding box|Gebiet, durch zwei Längen- und Breitengrade definiert, stellt Interessenobjekt dar (erkanntes Objekt)|
|Class-agnostic NMS|Bilderkennung, ohne eindeutig Bestimmte Klassenzugehörigkeit, meist für einfaches pre-processing genutzt|
|Confidence threshold|Wert 0-1 (wird prozentual gerechnet), bestimmt, ab wann eine potenzielle Erkennung als „erfolgreich“ deklariert und damit dem Nutzer angezeigt wird|
|Confusion matrix|Visualisiert Erkennungserfolge (Validierung) und setzt Metriken für mAP-Gleichung|
|Cosine learning rate scheduler|Kosinusfunktion, bestimmt, wie schnell der Trainingsvorgang in einen neuen Schritt (epoch, batch) übergeht, verhindert, dass sich das Modell unterm Optimum aufhängt|
|CUDA Device|Nummer der Grafikkarte, falls mehrere vorhanden sind und kein multi-GPU Training genutzt wird (0 für Standard)|
|CUDA|[Compute Unified Device Architecture](https://developer.nvidia.com/cuda-toolkit) = GPU fokussierte Programmumgebung („Programmiersprache der Grafikkarte“)|
|CuDNN|[CuDNN](https://developer.nvidia.com/cudnn) = CUDA Deep Neural Network|
|Early stopping patience|Wie viele epochs ohne Verbesserungen dürfen auftreten, bevor das Modelltraining abgebrochen wird|
|Epoch|Durchlauf durch gesamten Trainingsdatensatz|
|FP16 half-precision inference|Konvertierung des Modells zu binärem Fließkomma-Computerzahlenformat (Deklarierungen werden zu Näherungen), Präzision geht verloren|
|Global training seed|RNG Seed, welcher für jede Trainingseinheit genutzt wird (hilft beim Debuggen)|
|Gsutil|Google Cloud Storage Command-Paket, um Modelle und Zwischenstände automatisch in einer Cloud zu speichern|
|HPem|Hyperparameter evolution metadata, Anwendungsspezifische Konfigurationsmöglichkeiten vom Modelltraining|
|HPem anchor_t|Steuert Nummer an Detektionen relativ zur Performance (höherer Wert = weniger Detektionen, präziseres Modell)|
|HPem anchors|Nummer an Detektionen|
|HPem box|Box loss gain, offset Wert um dem box loss entgegenzusteuern|
|HPem cls|Cls loss gain, offset Wert um die performance, welche den cls loss zu verantworten hat zu steigern|
|HPem cls_pw|cls BCELoss positive_weight, Wert welcher die Balance der BCE Loss Funktion steuert (siehe unten)|
|HPem copy_paste|segment copy-paste probability, rcnn feature|
|HPem degrees|Bildrotation|
|HPem fl_gamma|Focal loss gamma, bestimmt wie gut das Modell auf zu viele „Hintergrundpixel“ reagieren soll (Anwendungsspezifisch)|
|HPem fliplr|flip left-right, dreht Bilder um dadurch mehr Trainingsdaten zu generieren|
|HPem flipud|flip up-down, dreht Bilder um dadurch mehr Trainingsdaten zu generieren|
|HPem hsv_h|Hsv_Hue augmentation, ändert Farbton|
|HPem hsv_s|Hsv_Saturation augmentation, ändert Farbsättigung|
|HPem hsv_v|Hsv_Value Augmentation, ändert Farbwerte zu z.B. BGR|
|HPem iou_t|Theoretisch dasselbe, wie der NMS IOU Threshhold, bloß für das Training und relativ irrelevant|
|HPem lr0|Learning-rate start basierend auf Cosine learning rate scheduler|
|HPem lrf|Learning-rate end, bestimmt das Ende des Trainings|
|HPem mixup|Bilineare Interpolation, Pixelvektoren von verschiedenen Bildern werden gemischt (robusteres Modell) (Formel unten)|
|HPem momentum|Bei der Optimierung des Trainingsvorganges werden Parameter oppositiv von den loss-Graphen aufgestellt, Momentum merkt sich dabei die ursprünglichen Parameter (falls Optimierung fehlschlägt)|
|HPem mosaic|Wählt zufällig Bilder aus um ein weiteres, neues Bild zu generieren (=Mosaik)|
|HPem obj|Obj loss gain, offset Wert um dem object loss entgegenzusteuern|
|HPem obj_pw|Obj BCELoss positive_weight, nutzt die gleiche Formel wie der cls loss um den object loss zu verringern (ist ein deklariertes Objekt wirklich eins?)|
|HPem perspective|Digitale, 3-dimensionale Perspektive wird auf das Bild angewandt|
|HPem scale|Hoch- oder Runterskalierung des Bildes|
|HPem shear|"Dreht" das Bild (ähnlich zu perspective)|
|HPem translate|Bewegt das eigentliche Bild in einem größeren Rahmen, um es danach aussehen zu lassen als wäre das Objekt immer an einer anderen Position|
|HPem warmup_bias_lr|Gibt Learning Rate in der warumup-phase an.|
|HPem warmup_epochs|Nummer an Iterationen, bei denen der Optimizer nicht auf voller Leistung läuft, um später im Trainingsvorgang Effektivität zu gewährleisten (bei uns 2 Epochs)|
|HPem warmup_momentum|Das gleiche wie warmup_epochs, aber ein prozentualer Wert, wodurch die warmup-phase durch die insgesamte Menge an epochs erhöht wird.|
|HPem weight_decay|Legt fest, wann und wie stark das Training vom Ursprungsmodell abweichen soll (sonst passiert es, dass das Modell auf neuen Daten nicht performt)|
|Hyperparameters|YOLOv5 hat ca. 30 Hyperparameter zur Optimierung des Modelltrainings bei spezifischen use-cases|
|inference|Anwendung des gelernten Modells auf neue, unbekannte Dateien (bei uns batch-wise inference)|
|inference size|Bildgröße, welche beim aktiven Bearbeiten des Modells genutzt wird|
|loss|Cls_loss, box_loss, obj_loss Stellt den Datenverlust (CUDA Leakage) während des Trainings dar, Werte sollten immer kleiner werden|
|mAP|Mean average prediction value, gibt im Training an wie „präzise“ das Modell ist (Zielwert: 0,75-0,95)|
|Max dataloader workers|Begrenzt Menge an Daten, die parallel für die Trainingsbearbeitung geladen wird (sonst wird CPU so stark wie möglich ausgelastet)|
|Multi-scale inference|Objekte können auf verschiedenen relativen Bildgrößen erkannt werden (Vordergrund/Hintergrund), wird vorgebeugt durch größere Trainingssätze|
|NMS IOU threshold|Non-Max-Suppression Intersection over Union, verhindert, dass ein Objekt öfter als einmal über den confidence threshold als Erkennung deklariert wird (über 0-1 Wert einstellbar, Wert Nutzungsabhängig)|
|Quad dataloader|Läd 4 Bilder gleichzeitig über einen worker, dadurch mehr Effizienz und weniger CPU-Auslastung (evtl. weniger Präzision)|
|Recall|Misst, wie gut Richtig Positive aus allen Erkennungen gefunden werden (auch negative Erkennungen)|
|Tensor|Multilineare Abbildung, die eine bestimmte Anzahl von Vektoren auf einen Vektor abbildet und eine universelle Eigenschaft erfüllt|
|TensorFlow|Framework zur datenstromorientieren Programmierung (maschinelles Lernen)|
|Video frame-rate stride|Limitiert die FPS des Input Videos auf einen Festen Wert|
|Visualize features|Darstellung von externen Features, wie z.B. eigene plotting Routinen (in Gesichtserkennung genutzt)|
|weights|„Modell.pt“ – Name des genutzten Modells im Root-Verzeichnis|
|x/lr0-2|3, sich während des Trainings ändernde Werte, welche darstellen, welches Modell der Multi-scale Inference genutzt wird (0-groß, 1-mittel, 2-klein)|
