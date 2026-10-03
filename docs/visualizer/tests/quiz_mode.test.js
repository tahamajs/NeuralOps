/**
 * tests/quiz_mode.test.js
 * =======================
 * Unit tests for Architecture Mastery Quiz Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { QUIZ_QUESTIONS, renderQuizWidget } from '../src/educational/quiz_mode.js';

describe('Architecture Mastery Quiz Engine', () => {
    it('verifies all 5 quiz questions have valid answers, options, and explanations', () => {
        assert.strictEqual(QUIZ_QUESTIONS.length, 5);

        QUIZ_QUESTIONS.forEach((q, idx) => {
            assert.ok(q.id);
            assert.ok(q.title);
            assert.ok(q.question);
            assert.ok(q.options.length >= 4, `Question ${idx} must have at least 4 options`);
            assert.ok(q.correctIndex >= 0 && q.correctIndex < q.options.length);
            assert.ok(q.explanation.length > 20, `Question ${idx} explanation must be descriptive`);
        });
    });

    it('renders quiz question into mock DOM container without throwing', () => {
        const mockContainer = {
            innerHTML: '',
            querySelectorAll: () => [],
            querySelector: () => null
        };

        renderQuizWidget(mockContainer);
        assert.ok(mockContainer.innerHTML.includes('quiz-widget'));
        assert.ok(mockContainer.innerHTML.includes('MASTERY QUIZ'));
        assert.ok(mockContainer.innerHTML.includes('Question 1 of 5'));
    });
});
