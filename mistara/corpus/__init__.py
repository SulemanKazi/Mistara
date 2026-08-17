"""Authenticated-corpus matching: the stage that verifies rather than reads.

For sacred text, OCR output is a *search query*, never the answer. The answer
comes from an authenticated corpus via alignment, which is why nothing in this
package calls a model.
"""
