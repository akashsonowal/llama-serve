{{/*
App name
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "helpers.name" -}}
{{- default .Chart.Name .Values.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified Release name. This includes the app name and the release.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "helpers.releaseName" -}}
{{- $name := default .Chart.Name .Values.name -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Chart Release.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "helpers.chartRelease" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Define DNS Name */}}
{{- define "helpers.dnsName" -}}
  {{- if .Values.hostname -}}
  {{- printf "%s" .Values.hostname -}}
  {{- else -}}
  {{- printf "%s-cluster-url .Release.Name -}}
  {{- end -}}
{{- end -}}