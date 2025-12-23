GHCR_USER = clarenceee
IMAGE_NAME = agentic-rag
GHCR_IMAGE = ghcr.io/$(GHCR_USER)/$(IMAGE_NAME)

.PHONY: run clean docker-build docker-run docker-push

run:
	@echo "Starting the app..."
	cd service && streamlit run main.py --server.runOnSave true

docker-build:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION environment variable is not set."; \
		exit 1; \
	fi
	@echo "Building Docker image with version: $(VERSION)"
	docker build -t $(GHCR_IMAGE):$(VERSION) -t $(GHCR_IMAGE):latest .

docker-run:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION environment variable is not set."; \
		exit 1; \
	fi
	docker run -p 8501:8501 \
	--add-host=host.docker.internal:host-gateway \
	--env-file ./service/.env \
	$(GHCR_IMAGE):$(VERSION)

docker-push:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION environment variable is not set."; \
		exit 1; \
	fi
	@if [ -z "$(CR_PAT)" ]; then \
		echo "Error: CR_PAT environment variable is not set. Please set your GitHub Container Registry token."; \
		exit 1; \
	fi
	echo "Pushing to GitHub Container Registry..."
	docker logout ghcr.io
	echo "$(CR_PAT)" | docker login ghcr.io -u $(GHCR_USER) --password-stdin
	@docker push $(GHCR_IMAGE):$(VERSION)
	@docker push $(GHCR_IMAGE):latest

clean:
	@echo "Cleaning up..."
	# rm -f myapp *.o

.DEFAULT_GOAL := run
