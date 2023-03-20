---
layout: page
permalink: /materials/
title: materials
description: Materials for courses I taught.
nav: true
nav_order: 3
display_categories: [neural, work, fun]
horizontal: false
---

<!-- pages/materials.md -->
<div class="projects">
{%- if site.enable_material_categories and page.display_categories %}
  <!-- Display categorized materials -->
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  {%- assign categorized_materials = site.materials | where: "category", category -%}
  {%- assign sorted_materials = categorized_materials | sort: "importance" %}
  <!-- Generate cards for each material -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for material in sorted_materials -%}
      {% include materials_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for material in sorted_materials -%}
      {% include materials.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
  {% endfor %}

{%- else -%}
<!-- Display materials without categories -->
  {%- assign sorted_materials = site.materials | sort: "importance" -%}
  <!-- Generate cards for each material -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for material in sorted_materials -%}
      {% include materials_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for material in sorted_materials -%}
      {% include materials.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
{%- endif -%}
</div>