IMAGE := sii-clip

build:
	docker build -t $(IMAGE) .


shell:
	docker run -it \
		--shm-size=24g \
		-e DISPLAY=:0 \
		-e QT_X11_NO_MITSHM=1 \
		-v ./:/opt/project \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		--rm \
		sii-clip /bin/bash

