# Nombre de la imagen Docker
IMAGE := sii-clip

# Carpeta del proyecto dentro del contenedor
WORKDIR := /opt/project

# Ruta local del proyecto
LOCALDIR := $(shell pwd)

# ----------------------------------------------------------------
# Construcción de la imagen
# ----------------------------------------------------------------
build:
	docker build -t $(IMAGE) .

# ----------------------------------------------------------------
# Shell interactivo
# ----------------------------------------------------------------
shell:
	docker run -it \
		--shm-size=24g \
		-e DISPLAY=:0 \
		-e QT_X11_NO_MITSHM=1 \
		-v $(LOCALDIR):$(WORKDIR) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		--rm \
		$(IMAGE) /bin/bash

# ----------------------------------------------------------------
# Recetas para ejecutar scripts
# ----------------------------------------------------------------
run-blip:
	docker run -it \
		--shm-size=24g \
		-v $(LOCALDIR):$(WORKDIR) \
		--rm \
		$(IMAGE) python $(WORKDIR)/blip.py

run-blip2:
	docker run -it \
		--shm-size=24g \
		-v $(LOCALDIR):$(WORKDIR) \
		--rm \
		$(IMAGE) python $(WORKDIR)/blip2.py

run-clip:
	docker run -it \
		--shm-size=24g \
		-v $(LOCALDIR):$(WORKDIR) \
		--rm \
		$(IMAGE) python $(WORKDIR)/clip.py

run-clip-laion:
	docker run -it \
		--shm-size=24g \
		-v $(LOCALDIR):$(WORKDIR) \
		--rm \
		$(IMAGE) python $(WORKDIR)/clip_laion.py

run-preclip:
	docker run -it \
		--shm-size=24g \
		-v $(LOCALDIR):$(WORKDIR) \
		--rm \
		$(IMAGE) python $(WORKDIR)/pre-clip.py

run-flava:
	docker run -it \
		--shm-size=24g \
		-v $(LOCALDIR):$(WORKDIR) \
		--rm \
		$(IMAGE) python $(WORKDIR)/flava.py

run-flava2:
	docker run -it \
		--shm-size=24g \
		-v $(LOCALDIR):$(WORKDIR) \
		--rm \
		$(IMAGE) python $(WORKDIR)/flava2.py

# ----------------------------------------------------------------
# Ayuda
# ----------------------------------------------------------------
help:
	@echo "make build        → Construir la imagen Docker"
	@echo "make shell        → Abrir un shell interactivo en el contenedor"
	@echo "make run-blip     → Ejecutar blip.py"
	@echo "make run-blip2    → Ejecutar blip2.py"
	@echo "make run-clip     → Ejecutar clip.py"
	@echo "make run-clip-laion → Ejecutar clip_laion.py"
	@echo "make run-preclip  → Ejecutar pre-clip.py"
	@echo "make run-flava    → Ejecutar flava.py"
	@echo "make run-flava2   → Ejecutar flava2.py"
