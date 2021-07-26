{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "ria.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ria.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ria.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "ria.labels" -}}
app.kubernetes.io/name: {{ include "ria.name" . }}
helm.sh/chart: {{ include "ria.chart" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Create a fully qualified configmap name.
*/}}
{{- define "service.configmap.fullname" -}}
{{- printf "%s-%s" .Release.Name .Release.Namespace -}}
{{- end -}}

{{/*
Service registry name.
*/}}
{{- define "service.registry.name" -}}
{{- printf "%s-%s-%s-%s" .Release.Name .Release.Namespace "docker" "registry" -}}
{{- end -}}

{{/*
Service registry secret.
*/}}
{{- define "service.registry.secret" -}}
{{- printf "{\"auths\": {\"%s\": {\"username\": \"%s\", \"password\": \"%s\", \"auth\": \"%s\"}}}" .Values.docker.registry .Values.docker.user .Values.docker.password (printf "%s:%s" .Values.docker.user .Values.docker.password | b64enc) | b64enc }}
{{- end -}}

{{/*
Service image name.
*/}}
{{- define "service.image.name" -}}
{{- printf "%s/%s/%s:%s" .Values.docker.registry .Values.docker.user .Values.docker.image .Values.docker.tag -}}
{{- end -}}