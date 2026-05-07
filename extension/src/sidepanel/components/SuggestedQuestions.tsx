/**
 * SuggestedQuestions — pill-shaped quick action buttons.
 */

interface SuggestedQuestionsProps {
  questions: string[];
  onSelect: (question: string) => void;
  compact?: boolean;
}

export default function SuggestedQuestions({
  questions,
  onSelect,
  compact = false,
}: SuggestedQuestionsProps) {
  if (!questions || questions.length === 0) return null;

  return (
    <div
      className={`flex flex-wrap gap-2 ${
        compact ? 'justify-start' : 'justify-center max-w-[280px]'
      }`}
    >
      {questions.map((question, idx) => (
        <button
          key={idx}
          onClick={() => onSelect(question)}
          className="suggestion-pill"
          id={`suggestion-${idx}`}
        >
          {question}
        </button>
      ))}
    </div>
  );
}
