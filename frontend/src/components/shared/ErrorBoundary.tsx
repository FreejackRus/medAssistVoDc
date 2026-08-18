import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Render error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 text-center">
          <p className="text-lg font-medium">Что-то пошло не так</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Попробуйте перезагрузить страницу
          </p>
          <button
            className="mt-4 rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground"
            onClick={() => this.setState({ hasError: false })}
          >
            Попробовать снова
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
